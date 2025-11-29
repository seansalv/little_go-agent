"""Legacy entry point preserved for autograders."""

from go_agent_api.go_engine import pick_move
from go_agent_api.go_engine.engine import main

__all__ = ["pick_move", "main"]


if __name__ == "__main__":
	main()
