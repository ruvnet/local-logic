from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

def init_phoenix(service_name="poker-bot"):
    """Initialize Phoenix tracing"""
    try:
        # Get configuration from environment with defaults
        phoenix_host = os.getenv('PHOENIX_HOST', 'phoenix')
        phoenix_port = os.getenv('PHOENIX_GRPC_PORT', '4317')
        endpoint = f"http://{phoenix_host}:{phoenix_port}"
        
        print(f"Initializing Phoenix tracing for {service_name}")
        print(f"Using endpoint: {endpoint}")
        
        # Create OTLP exporter without retry parameter
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            timeout=30  # 30 second timeout
        )
        
        # Create TracerProvider with resource attributes
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(
            otlp_exporter,
            max_export_batch_size=512,
            schedule_delay_millis=5000
        ))
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        print("Phoenix tracing initialized successfully")
        
        return tracer_provider
    except Exception as e:
        print(f"Error: Failed to initialize Phoenix tracing:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        print("Continuing without tracing...")
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
