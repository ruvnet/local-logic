from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

def init_phoenix(service_name="poker-bot"):
    """Initialize Phoenix tracing"""
    try:
        # Create OTLP exporter
        phoenix_host = os.getenv('PHOENIX_HOST', 'phoenix')
        phoenix_port = os.getenv('PHOENIX_GRPC_PORT', '4317')
        endpoint = f"http://{phoenix_host}:{phoenix_port}"
        
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        
        # Create TracerProvider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        return tracer_provider
    except Exception as e:
        print(f"Warning: Failed to initialize Phoenix: {str(e)}")
        return None

def get_tracer(name):
    """Get a tracer instance"""
    return trace.get_tracer(name)
