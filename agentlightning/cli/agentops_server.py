import time
from agentlightning.instrumentation.agentops import AgentOpsServerManager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start AgentOps server")
    parser.add_argument("--daemon", action="store_true", help="Run server as a daemon")
    parser.add_argument("--port", type=int, default=8002, help="Port to run the server on")
    args = parser.parse_args()

    manager = AgentOpsServerManager(daemon=args.daemon, port=args.port)
    try:
        manager.start()
        # Wait forever
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
