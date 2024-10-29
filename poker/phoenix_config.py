from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os
import logging
import dspy

def init_phoenix(service_name="poker-bot"):
    """Initialize Phoenix tracing"""
    try:
        # Suppress OpenTelemetry logging
        logging.getLogger('opentelemetry').setLevel(logging.WARNING)
        
        # Get configuration from environment with defaults
        phoenix_host = os.getenv('PHOENIX_HOST', 'localhost')
        phoenix_port = os.getenv('PHOENIX_GRPC_PORT', '4317')
        endpoint = f"http://{phoenix_host}:{phoenix_port}"
        
        print(f"Initializing Phoenix tracing with endpoint: {endpoint}")
        
        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=True
        )
        
        # Create and configure TracerProvider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Initialize DSPy instrumentation
        try:
            from openinference.instrumentation.dspy import DSPyInstrumentor
            instrumentor = DSPyInstrumentor()
            instrumentor.instrument(tracer_provider=tracer_provider)
            
            # Configure DSPy to use the tracer
            dspy.settings.configure(
                trace_provider=tracer_provider,
                trace_enabled=True
            )
            
        except ImportError as e:
            print(f"Warning: Could not initialize DSPy instrumentation: {str(e)}")
            print("Phoenix tracing will continue without DSPy integration")
            
        print("Phoenix optimization initialized successfully")
        return tracer_provider
        
    except Exception as e:
        print(f"Error initializing Phoenix tracing: {str(e)}")
        return None

def get_tracer(name):
    """Get a tracer instance with error handling"""
    try:
        return trace.get_tracer(name)
    except Exception as e:
        print(f"Error getting tracer '{name}': {str(e)}")
        # Return a dummy tracer that won't break the application
        class DummyTracer:
            def start_as_current_span(self, *args, **kwargs):
                class DummySpan:
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def set_attribute(self, *args): pass
                    def record_exception(self, *args): pass
                return DummySpan()
        return DummyTracer()
