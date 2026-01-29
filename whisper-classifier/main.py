#!/usr/bin/env python3
"""
Real-Time Voice Transcription & Coding Question Classifier

Main entry point for the application.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.utils import load_config, TranscriptionResult, get_audio_devices, format_duration
from src.pipeline import RealTimePipeline
from src.audio_capture import list_audio_devices


# Global state for signal handling
pipeline: Optional[RealTimePipeline] = None
running = True


def signal_handler(sig, frame):
    """Handle interrupt signal."""
    global running
    running = False
    print("\n\n🛑 Stopping...")


def print_header():
    """Print application header."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]🎙️ Real-Time Voice Transcription & Classifier[/bold cyan]\n"
            "[dim]Speak into your microphone - results appear in real-time[/dim]",
            border_style="cyan"
        ))
    else:
        print("=" * 60)
        print("🎙️ Real-Time Voice Transcription & Classifier")
        print("Speak into your microphone - results appear in real-time")
        print("=" * 60)


def print_result_rich(result: TranscriptionResult, console: Console):
    """Print result using rich formatting."""
    # Color based on classification
    if result.classification == "CODING":
        class_style = "bold green"
        class_emoji = "💻"
    else:
        class_style = "bold yellow"
        class_emoji = "💬"
    
    # Format latency
    if result.latency_ms < 300:
        latency_style = "green"
    elif result.latency_ms < 500:
        latency_style = "yellow"
    else:
        latency_style = "red"
    
    # Create output
    text = Text()
    text.append(f"\n{class_emoji} ", style="bold")
    text.append(f"[{result.classification}]", style=class_style)
    text.append(f" ({result.confidence:.0%})", style="dim")
    text.append(f" • ", style="dim")
    text.append(f"{result.latency_ms:.0f}ms", style=latency_style)
    
    console.print(text)
    console.print(f"   \"{result.transcription}\"", style="white")


def print_result_simple(result: TranscriptionResult):
    """Print result using simple formatting."""
    class_emoji = "💻" if result.classification == "CODING" else "💬"
    
    print(f"\n{class_emoji} [{result.classification}] ({result.confidence:.0%}) • {result.latency_ms:.0f}ms")
    print(f"   \"{result.transcription}\"")


def print_stats(pipeline: RealTimePipeline, duration: float):
    """Print final statistics."""
    stats = pipeline.get_stats()
    
    if RICH_AVAILABLE:
        console = Console()
        
        table = Table(title="📊 Session Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Duration", format_duration(duration))
        table.add_row("Total Processed", str(stats['total_processed']))
        table.add_row("Coding Questions", str(stats['coding_count']))
        table.add_row("Non-Coding Questions", str(stats['non_coding_count']))
        table.add_row("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
        table.add_row("Min Latency", f"{stats['min_latency_ms']:.0f}ms")
        table.add_row("Max Latency", f"{stats['max_latency_ms']:.0f}ms")
        
        console.print("\n")
        console.print(table)
    else:
        print("\n" + "=" * 40)
        print("📊 Session Statistics")
        print("=" * 40)
        print(f"Duration: {format_duration(duration)}")
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Coding Questions: {stats['coding_count']}")
        print(f"Non-Coding Questions: {stats['non_coding_count']}")
        print(f"Avg Latency: {stats['avg_latency_ms']:.0f}ms")
        print(f"Min Latency: {stats['min_latency_ms']:.0f}ms")
        print(f"Max Latency: {stats['max_latency_ms']:.0f}ms")


def main():
    """Main entry point."""
    global pipeline, running
    
    parser = argparse.ArgumentParser(
        description="Real-Time Voice Transcription & Coding Question Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --model medium     # Use medium Whisper model
  python main.py --device 1         # Use specific audio device
  python main.py --list-devices     # List available audio devices
  python main.py --no-color         # Disable colored output
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        default=None,
        help='Whisper model size (overrides config)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=None,
        help='Audio input device index'
    )
    
    parser.add_argument(
        '--list-devices', '-l',
        action='store_true',
        help='List available audio devices and exit'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log transcriptions to file (JSONL format)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        help='Language code (e.g., en, es, fr) or auto for detection'
    )
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return 0
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"⚠️ Config not found, using defaults: {e}")
        config = {
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_duration': 1.5,
                'vad_aggressiveness': 2,
                'silence_threshold': 0.5,
                'min_speech_duration': 0.3
            },
            'transcription': {
                'model_size': 'small',
                'device': 'auto',
                'compute_type': 'int8',
                'language': 'en',
                'beam_size': 5
            },
            'classification': {},
            'text_cleanup': {
                'remove_filler_words': True,
                'filler_words': ['um', 'umm', 'uh', 'uhh', 'like', 'you know']
            },
            'pipeline': {
                'enable_logging': False
            }
        }
    
    # Override config with command line arguments
    if args.model:
        config['transcription']['model_size'] = args.model
    
    if args.device is not None:
        config['audio']['device'] = args.device
    
    if args.language:
        config['transcription']['language'] = None if args.language == 'auto' else args.language
    
    if args.log_file:
        config['pipeline']['enable_logging'] = True
        config['pipeline']['log_file'] = args.log_file
    
    if args.json:
        config['output'] = config.get('output', {})
        config['output']['json_output'] = True
    
    # Setup console
    use_rich = RICH_AVAILABLE and not args.no_color and not args.json
    console = Console() if use_rich else None
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print header
    if not args.json:
        print_header()
        print()
    
    # Create and start pipeline
    try:
        if not args.json:
            print("🔄 Initializing pipeline...")
        
        pipeline = RealTimePipeline(config)
        
        # Set result callback
        def on_result(result: TranscriptionResult):
            if args.json:
                print(result.to_json(), flush=True)
            elif use_rich:
                print_result_rich(result, console)
            else:
                print_result_simple(result)
        
        pipeline.set_result_callback(on_result)
        
        if not args.json:
            print("✅ Pipeline ready!")
            print("\n🎤 Listening... (Press Ctrl+C to stop)\n")
            print("-" * 50)
        
        # Start pipeline
        start_time = time.time()
        pipeline.start()
        
        # Main loop
        while running:
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    finally:
        # Stop pipeline
        if pipeline:
            pipeline.stop()
            duration = time.time() - start_time
            
            if not args.json:
                print_stats(pipeline, duration)
    
    if not args.json:
        print("\n👋 Goodbye!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
