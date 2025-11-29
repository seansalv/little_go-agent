"use client";

import Link from "next/link";
import { useCallback, useMemo, useRef, useState } from "react";

const BOARD_SIZE = 5;
const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;
const API_URL = process.env.NEXT_PUBLIC_GO_AGENT_API_URL ?? "http://localhost:8000/move";
const COLUMN_LABELS = ["A", "B", "C", "D", "E"];
const ROW_LABELS = ["1", "2", "3", "4", "5"];

type Board = number[][];
type Point = { row: number; col: number };
type MovePayload = Point | null;

const buildBoard = (): Board =>
	Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(EMPTY));

const cloneBoard = (board: Board): Board => board.map((row) => [...row]);

const neighbors = (row: number, col: number): Point[] => {
	const deltas = [
		[1, 0],
		[-1, 0],
		[0, 1],
		[0, -1],
	];
	return deltas
		.map(([dr, dc]) => ({ row: row + dr, col: col + dc }))
		.filter(({ row: r, col: c }) => r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE);
};

const serialize = ({ row, col }: Point) => `${row}:${col}`;

const collectGroup = (board: Board, row: number, col: number): Point[] => {
	const color = board[row][col];
	if (color === EMPTY) {
		return [];
	}
	const stack: Point[] = [{ row, col }];
	const visited = new Set([serialize({ row, col })]);
	const group: Point[] = [];

	while (stack.length) {
		const current = stack.pop()!;
		group.push(current);
		for (const n of neighbors(current.row, current.col)) {
			if (board[n.row][n.col] === color && !visited.has(serialize(n))) {
				visited.add(serialize(n));
				stack.push(n);
			}
		}
	}
	return group;
};

const countLiberties = (board: Board, group: Point[]): number => {
	const liberties = new Set<string>();
	group.forEach(({ row, col }) => {
		neighbors(row, col).forEach((n) => {
			if (board[n.row][n.col] === EMPTY) {
				liberties.add(serialize(n));
			}
		});
	});
	return liberties.size;
};

const removeGroup = (board: Board, group: Point[]): void => {
	group.forEach(({ row, col }) => {
		board[row][col] = EMPTY;
	});
};

const applyMove = (board: Board, row: number, col: number, color: number): Board | null => {
	if (board[row][col] !== EMPTY) {
		return null;
	}
	const snapshot = cloneBoard(board);
	snapshot[row][col] = color;
	const opponent = color === BLACK ? WHITE : BLACK;
	let captured = false;

	for (const n of neighbors(row, col)) {
		if (snapshot[n.row][n.col] === opponent) {
			const group = collectGroup(snapshot, n.row, n.col);
			if (group.length && countLiberties(snapshot, group) === 0) {
				removeGroup(snapshot, group);
				captured = true;
			}
		}
	}

	const myGroup = collectGroup(snapshot, row, col);
	const myLiberties = countLiberties(snapshot, myGroup);
	if (myLiberties === 0 && !captured) {
		return null; // suicide
	}
	return snapshot;
};

const boardsEqual = (a: Board | null, b: Board | null): boolean => {
	if (a === b) return true;
	if (!a || !b) return false;
	for (let i = 0; i < BOARD_SIZE; i += 1) {
		for (let j = 0; j < BOARD_SIZE; j += 1) {
			if (a[i][j] !== b[i][j]) {
				return false;
			}
		}
	}
	return true;
};


export default function GoAgentPage() {
	const [currentBoard, setCurrentBoard] = useState<Board>(() => buildBoard());
	const [previousBoard, setPreviousBoard] = useState<Board | null>(null);
	const [moveLog, setMoveLog] = useState<string[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [status, setStatus] = useState<"active" | "finished">("active");
	const [, setConsecutivePasses] = useState(0);
	const [lastMove, setLastMove] = useState<Point | null>(null);

	const lastRequest = useRef<{ prev: Board | null; curr: Board } | null>(null);

	const apiHint = useMemo(() => API_URL.replace(/^https?:\/\//, ""), []);

	const pushLog = useCallback((entry: string) => {
		setMoveLog((history) => [...history, entry]);
	}, []);

	const resetGame = () => {
		setCurrentBoard(buildBoard());
		setPreviousBoard(null);
		setMoveLog([]);
		setIsLoading(false);
		setError(null);
		setStatus("active");
		setConsecutivePasses(0);
		setLastMove(null);
		lastRequest.current = null;
	};

	const resolveStatusAfterPass = useCallback(() => {
		setConsecutivePasses((count) => {
			const next = count + 1;
			if (next >= 2) {
				setStatus("finished");
				pushLog("Scoring • both players passed");
			}
			return next;
		});
	}, [pushLog]);

	const handleBotMove = useCallback(
		(move: MovePayload, boardBeforeBot: Board) => {
			if (move === null) {
				pushLog("White • PASS");
				setPreviousBoard(cloneBoard(boardBeforeBot));
				setCurrentBoard(cloneBoard(boardBeforeBot));
				setLastMove(null);
				resolveStatusAfterPass();
				return;
			}
			const nextBoard = applyMove(boardBeforeBot, move.row, move.col, WHITE);
			if (!nextBoard) {
				setError("Agent produced an illegal move. Please restart the game.");
				setStatus("finished");
				return;
			}
			setPreviousBoard(cloneBoard(boardBeforeBot));
			setCurrentBoard(nextBoard);
			pushLog(`White • (${move.row + 1}, ${move.col + 1})`);
			setLastMove(move);
			setConsecutivePasses(0);
		},
		[pushLog, resolveStatusAfterPass],
	);

	const sendRequest = useCallback(
		async (prevSnapshot: Board | null, currentSnapshot: Board) => {
			lastRequest.current = {
				prev: prevSnapshot ? cloneBoard(prevSnapshot) : null,
				curr: cloneBoard(currentSnapshot),
			};
			setIsLoading(true);
			setError(null);

			const controller = new AbortController();
			const timer = window.setTimeout(() => controller.abort(), 12000);
			try {
				const response = await fetch(API_URL, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						player: WHITE,
						previousBoard: prevSnapshot,
						currentBoard: currentSnapshot,
					}),
					signal: controller.signal,
				});
				if (!response.ok) {
					const detail = await response.text();
					throw new Error(detail || `Engine responded with ${response.status}`);
				}
				const data = (await response.json()) as { move: MovePayload };
				handleBotMove(data.move, currentSnapshot);
			} catch (err) {
				if ((err as Error).name === "AbortError") {
					setError("Engine timed out. You can retry once it wakes up.");
				} else {
					setError((err as Error).message || "Something went wrong.");
				}
			} finally {
				window.clearTimeout(timer);
				setIsLoading(false);
			}
		},
		[handleBotMove],
	);

	const retryLast = () => {
		if (!lastRequest.current || isLoading) return;
		const { prev, curr } = lastRequest.current;
		sendRequest(prev ? cloneBoard(prev) : null, cloneBoard(curr));
	};

	const placeStone = (row: number, col: number) => {
		if (status === "finished" || isLoading) return;
		setError(null);
		const beforeMove = cloneBoard(currentBoard);
		const nextBoard = applyMove(beforeMove, row, col, BLACK);
		if (!nextBoard) {
			setError("Illegal move. Try another intersection.");
			return;
		}
		if (previousBoard && boardsEqual(previousBoard, nextBoard)) {
			setError("Simple KO: you cannot recreate the previous position.");
			return;
		}
		pushLog(`Black • (${row + 1}, ${col + 1})`);
		setPreviousBoard(beforeMove);
		setCurrentBoard(nextBoard);
		setLastMove({ row, col });
		setConsecutivePasses(0);
		sendRequest(beforeMove, nextBoard);
	};

	const forcePass = () => {
		if (status === "finished" || isLoading) return;
		pushLog("Black • PASS");
		const snapshot = cloneBoard(currentBoard);
		setPreviousBoard(snapshot);
		setLastMove(null);
		resolveStatusAfterPass();
		sendRequest(snapshot, snapshot);
	};

	const boardCells = currentBoard.map((row, rowIdx) =>
		row.map((value, colIdx) => {
			const isLast = lastMove && lastMove.row === rowIdx && lastMove.col === colIdx;
			return (
				<button
					key={`${rowIdx}-${colIdx}`}
					className="relative flex h-16 w-16 items-center justify-center border border-slate-700 bg-slate-900/70 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
					onClick={() => placeStone(rowIdx, colIdx)}
					disabled={status === "finished" || isLoading || value !== EMPTY}
					aria-label={`Row ${rowIdx + 1}, Column ${colIdx + 1}`}
				>
					{value !== EMPTY && (
						<span
							className={`h-11 w-11 rounded-full shadow-inner ${
								value === BLACK ? "bg-slate-50 shadow-slate-900/60" : "bg-amber-200 shadow-amber-900/50"
							}`}
						/>
					)}
					{isLast && (
						<span className="absolute h-3 w-3 rounded-full border border-slate-800 bg-sky-300 shadow shadow-sky-900/40" />
					)}
				</button>
			);
		}),
	);

	return (
		<div className="min-h-screen bg-slate-950 text-slate-100">
			<div className="mx-auto flex max-w-6xl flex-col gap-10 px-4 py-10 sm:px-8">
				<header className="space-y-4">
					<p className="text-sm uppercase tracking-[0.25em] text-sky-400">Project</p>
					<h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
						Play against the Little Go alpha-beta bot
					</h1>
					<p className="max-w-3xl text-lg text-slate-300">
						The same search that powers my competition bot now lives behind a FastAPI service.
						Place stones on the 5×5 board, enforce basic KO locally, and let the agent respond in real
						time.
					</p>
					<div className="flex flex-wrap gap-4 text-sm text-slate-400">
						<span className="rounded-full border border-slate-800/80 px-3 py-1">
							Endpoint · {apiHint}
						</span>
						<span className="rounded-full border border-slate-800/80 px-3 py-1">You = Black</span>
						<span className="rounded-full border border-slate-800/80 px-3 py-1">Bot = White</span>
						<Link
							href="https://github.com/seansalv/little-go-agent"
							target="_blank"
							className="rounded-full border border-sky-500/40 px-3 py-1 text-sky-300 underline-offset-4 hover:border-sky-300 hover:text-sky-200"
						>
							GitHub repo ↗
						</Link>
					</div>
				</header>

				<div className="grid gap-8 lg:grid-cols-[2fr,1fr]">
					<section className="space-y-6 rounded-3xl border border-slate-800/70 bg-slate-900/60 p-6 shadow-2xl shadow-black/40">
						<div className="flex items-center justify-between text-sm text-slate-400">
							<span>Status: {status === "finished" ? "Game over" : isLoading ? "Agent thinking…" : "Your move"}</span>
							{error && (
								<button
									onClick={retryLast}
									className="font-medium text-sky-300 underline-offset-4 hover:underline disabled:opacity-50"
									disabled={!lastRequest.current || isLoading}
								>
									Retry last call
								</button>
							)}
						</div>
						<div className="space-y-3">
							<div className="ml-10 flex justify-between px-3 text-xs font-semibold uppercase tracking-[0.35em] text-slate-500">
								{COLUMN_LABELS.map((label) => (
									<span key={label}>{label}</span>
								))}
							</div>
							<div className="flex">
								<div className="mr-2 flex flex-col justify-between py-3 text-xs font-semibold text-slate-500">
									{ROW_LABELS.map((label) => (
										<span key={label}>{label}</span>
									))}
								</div>
								<div className="grid grid-cols-5 gap-1 rounded-2xl border border-slate-700 bg-gradient-to-br from-slate-900 to-slate-950 p-3">
									{boardCells}
								</div>
							</div>
						</div>
						{error && (
							<div className="rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
								{error}
							</div>
						)}
						<div className="flex flex-wrap gap-3">
							<button
								onClick={resetGame}
								className="rounded-full border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-sky-400 hover:text-sky-200 disabled:opacity-40"
							>
								Restart game
							</button>
							<button
								onClick={forcePass}
								disabled={status === "finished" || isLoading}
								className="rounded-full border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-sky-300 hover:text-sky-200 disabled:opacity-40"
							>
								Force pass
							</button>
						</div>
					</section>

					<aside className="flex flex-col gap-4 rounded-3xl border border-slate-800/70 bg-slate-900/40 p-6">
						<div className="flex items-center justify-between">
							<h2 className="text-lg font-semibold text-white">Move log</h2>
							<Link
								href="https://github.com/seansalv/little-go-agent"
								target="_blank"
								className="text-sm text-sky-300 underline-offset-4 hover:underline"
							>
								GitHub repo
							</Link>
						</div>
						<div className="h-64 overflow-y-auto rounded-2xl border border-slate-800/80 bg-slate-950/60 p-4 text-sm text-slate-200">
							{moveLog.length === 0 ? (
								<p className="text-slate-500">No moves yet. Black goes first.</p>
							) : (
								<ul className="space-y-2">
									{moveLog.map((entry, idx) => (
										<li key={`${entry}-${idx}`} className="flex items-center gap-3">
											<span className="text-xs text-slate-500">#{idx + 1}</span>
											<span>{entry}</span>
										</li>
									))}
								</ul>
							)}
						</div>
						<p className="text-xs text-slate-500">
							The UI mirrors the FastAPI validation rules. We compare boards locally to block simple KO
							and surface API errors if Render idles the container.
						</p>
					</aside>
				</div>
			</div>
		</div>
	);
}

