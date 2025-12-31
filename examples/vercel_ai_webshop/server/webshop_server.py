#!/usr/bin/env python3
"""
WebShop Flask Server

A lightweight Flask server that wraps the WebShop OpenAI Gym environment,
exposing /reset and /step HTTP endpoints for the Next.js frontend.

Usage:
    python webshop_server.py [--port PORT] [--num-products NUM]

Requires:
    - WebShop installed (run setup.sh first)
    - Python 3.8+
"""

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Session configuration
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
DEFAULT_NUM_PRODUCTS = 1000  # Small dataset for fast startup

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend


@dataclass
class Session:
    """Represents a WebShop session with its gym environment."""
    id: str
    env: Any  # gym.Env
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    instruction: str = ""


class SessionManager:
    """Manages WebShop gym environment sessions."""

    def __init__(self, num_products: int = DEFAULT_NUM_PRODUCTS):
        self.num_products = num_products
        self.sessions: Dict[str, Session] = {}
        self.lock = Lock()
        self._gym_registered = False

    def _ensure_gym_registered(self):
        """Lazily register the gym environment."""
        if self._gym_registered:
            return

        try:
            import gym
            # Import WebShop environment - this registers it with gym
            from web_agent_site.envs import WebAgentTextEnv
            self._gym_registered = True
            logger.info("WebShop gym environment registered successfully")
        except ImportError as e:
            logger.error(f"Failed to import WebShop: {e}")
            logger.error("Run setup.sh first to install WebShop")
            raise

    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_accessed > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            logger.info(f"Cleaning up expired session: {sid}")
            del self.sessions[sid]

    def create_session(self, session_id: Optional[str] = None,
                       instruction: Optional[str] = None) -> Session:
        """Create a new session with a fresh gym environment."""
        import gym

        self._ensure_gym_registered()
        self._cleanup_expired()

        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"

        # Create gym environment
        env = gym.make(
            'WebAgentTextEnv-v0',
            observation_mode='text',
            num_products=self.num_products
        )

        # Reset environment
        if instruction:
            obs = env.reset(instruction_text=instruction)
        else:
            obs = env.reset()

        instruction_text = env.get_instruction_text() if hasattr(env, 'get_instruction_text') else ""

        session = Session(
            id=sid,
            env=env,
            instruction=instruction_text
        )

        with self.lock:
            self.sessions[sid] = session

        logger.info(f"Created session {sid} with {self.num_products} products")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, updating last accessed time."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = time.time()
            return session

    def reset_session(self, session_id: str,
                      instruction: Optional[str] = None) -> Optional[Session]:
        """Reset an existing session or create a new one."""
        session = self.get_session(session_id)

        if session:
            # Reset existing environment
            if instruction:
                session.env.reset(instruction_text=instruction)
            else:
                session.env.reset()

            if hasattr(session.env, 'get_instruction_text'):
                session.instruction = session.env.get_instruction_text()

            logger.info(f"Reset existing session {session_id}")
            return session
        else:
            # Create new session
            return self.create_session(session_id, instruction)


# Global session manager (initialized on first request)
session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global session_manager
    if session_manager is None:
        num_products = int(os.environ.get('WEBSHOP_NUM_PRODUCTS', DEFAULT_NUM_PRODUCTS))
        session_manager = SessionManager(num_products=num_products)
    return session_manager


@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset or create a WebShop session.

    Request body:
        {
            "session_id": "optional_session_id",
            "instruction": "optional custom instruction"
        }

    Response:
        {
            "session_id": "session_xxx",
            "observation": "WebShop [SEP] Instruction: ...",
            "done": false,
            "reward": 0
        }
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id') or data.get('sessionId')
        instruction = data.get('instruction')

        manager = get_session_manager()

        if session_id:
            session = manager.reset_session(session_id, instruction)
        else:
            session = manager.create_session(instruction=instruction)

        if session is None:
            return jsonify({'error': 'Failed to create session'}), 500

        # Get initial observation
        env = session.env
        obs = env.observation if hasattr(env, 'observation') else ""

        # Try to get the current observation from the environment state
        if hasattr(env, 'state') and hasattr(env.state, 'get'):
            obs = env.state.get('observation', obs)

        # Construct observation text with instruction
        if session.instruction and session.instruction not in obs:
            obs = f"WebShop [SEP] Instruction: {session.instruction} [SEP] {obs}"

        return jsonify({
            'session_id': session.id,
            'observation': obs,
            'done': False,
            'reward': 0
        })

    except Exception as e:
        logger.exception("Error in /reset")
        return jsonify({'error': str(e)}), 500


@app.route('/step', methods=['POST'])
def step():
    """
    Execute an action in the WebShop environment.

    Request body:
        {
            "session_id": "session_xxx",
            "action": "search[red t-shirt]"
        }

    Response:
        {
            "session_id": "session_xxx",
            "observation": "[Back to Search] ...",
            "done": false,
            "reward": 0.0
        }
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id') or data.get('sessionId')
        action = data.get('action', '')

        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400

        if not action:
            return jsonify({'error': 'action is required'}), 400

        manager = get_session_manager()
        session = manager.get_session(session_id)

        if session is None:
            return jsonify({'error': f'Session {session_id} not found'}), 404

        # Execute action in gym environment
        obs, reward, done, info = session.env.step(action)

        logger.info(f"Session {session_id}: action={action}, reward={reward}, done={done}")

        return jsonify({
            'session_id': session.id,
            'observation': obs,
            'done': done,
            'reward': reward,
            'info': info if isinstance(info, dict) else {}
        })

    except Exception as e:
        logger.exception("Error in /step")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'sessions': len(get_session_manager().sessions) if session_manager else 0
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info."""
    return jsonify({
        'name': 'WebShop API Server',
        'version': '1.0.0',
        'endpoints': {
            'POST /reset': 'Create or reset a session',
            'POST /step': 'Execute an action',
            'GET /health': 'Health check'
        }
    })


def main():
    parser = argparse.ArgumentParser(description='WebShop Flask Server')
    parser.add_argument('--port', type=int, default=3000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--num-products', type=int, default=DEFAULT_NUM_PRODUCTS,
                        help='Number of products to load (default: 1000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Set environment variable for session manager
    os.environ['WEBSHOP_NUM_PRODUCTS'] = str(args.num_products)

    logger.info(f"Starting WebShop server on {args.host}:{args.port}")
    logger.info(f"Loading {args.num_products} products...")

    # Pre-initialize session manager to load data
    try:
        manager = get_session_manager()
        # Create a test session to trigger loading
        test_session = manager.create_session()
        logger.info("WebShop environment initialized successfully")
        # Clean up test session
        del manager.sessions[test_session.id]
    except Exception as e:
        logger.error(f"Failed to initialize WebShop: {e}")
        logger.error("Run setup.sh first to install WebShop and download data")
        sys.exit(1)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
