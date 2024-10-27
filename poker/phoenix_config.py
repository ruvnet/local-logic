from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

def init_phoenix(service_name="poker-bot"):
    """Initialize Phoenix tracing"""
    try:
        # Get configuration from environment with defaults
        phoenix_host = os.getenv('PHOENIX_HOST', 'localhost')
        phoenix_port = os.getenv('PHOENIX_GRPC_PORT', '4317')
        endpoint = f"http://{phoenix_host}:{phoenix_port}"
        
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
            DSPyInstrumentor().instrument()
        except ImportError:
            print("Warning: DSPy instrumentation not available")
            
        print(f"Phoenix tracing initialized successfully at {endpoint}")
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
                return DummySpan()
        return DummyTracer()
