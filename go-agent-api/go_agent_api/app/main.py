from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from go_agent_api.go_engine import BLACK, EMPTY, N, WHITE, env_time_limit, pick_move

ALLOWED_ORIGINS = [
	"https://seansalv.dev",
	"http://localhost:3000",
	"http://127.0.0.1:3000",
]
Board = List[List[int]]


class Position(BaseModel):
	row: int = Field(..., ge=0, lt=N)
	col: int = Field(..., ge=0, lt=N)


class MoveRequest(BaseModel):
	player: int = Field(..., description="1 = black, 2 = white")
	previousBoard: Optional[Board] = Field(default=None)
	currentBoard: Board = Field(..., min_length=N, max_length=N)


class MoveResponse(BaseModel):
	move: Optional[Position]


def _ensure_board(name: str, board: Optional[Board], *, required: bool) -> Optional[Board]:
	if board is None:
		if required:
			raise HTTPException(status_code=422, detail=f"{name} is required")
		return None
	if len(board) != N:
		raise HTTPException(status_code=422, detail=f"{name} must have {N} rows")
	canonical: Board = []
	for row_idx, row in enumerate(board):
		if len(row) != N:
			raise HTTPException(
				status_code=422,
				detail=f"{name} row {row_idx} must have {N} columns",
			)
		new_row = []
		for col_idx, value in enumerate(row):
			try:
				cell = int(value)
			except (TypeError, ValueError) as exc:
				raise HTTPException(
					status_code=422,
					detail=f"{name} row {row_idx} col {col_idx} must be an integer",
				) from exc
			if cell not in (EMPTY, BLACK, WHITE):
				raise HTTPException(
					status_code=422,
					detail=f"{name} values must be 0 (empty), 1 (black), or 2 (white)",
				)
			new_row.append(cell)
		canonical.append(new_row)
	return canonical


def _board_has_stones(board: Board) -> bool:
	return any(cell != EMPTY for row in board for cell in row)


app = FastAPI(
	title="Little Go Alpha-Beta API",
	description="Serve alpha-beta moves for a 5x5 Go board.",
	version="0.1.0",
)
app.add_middleware(
	CORSMiddleware,
	allow_origins=ALLOWED_ORIGINS,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/move", response_model=MoveResponse, tags=["gameplay"])
async def get_move(payload: MoveRequest) -> MoveResponse:
	if payload.player not in (BLACK, WHITE):
		raise HTTPException(status_code=422, detail="player must be 1 (black) or 2 (white)")
	prev_board = _ensure_board("previousBoard", payload.previousBoard, required=False)
	cur_board = _ensure_board("currentBoard", payload.currentBoard, required=True)
	if prev_board is None and _board_has_stones(cur_board):
		raise HTTPException(
			status_code=422,
			detail="previousBoard is required once either player has placed a stone",
		)

	time_limit = env_time_limit()
	try:
		engine_move = await asyncio.to_thread(
			pick_move,
			prev_board,
			cur_board,
			payload.player,
			time_limit,
		)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	if engine_move is None:
		return MoveResponse(move=None)
	row, col = engine_move
	return MoveResponse(move=Position(row=row, col=col))


