#!/usr/bin/env python3
# my_player3.py  — Alpha-Beta engine with TT, PV, Killers, History, Aspiration, Quiescence
# (Python 3.7.5 compatible)

import sys, time, random
from copy import deepcopy

N = 5
EMPTY, BLACK, WHITE = 0, 1, 2
KOMI = 2.5

# ---------- I/O ----------
def read_input(path="input.txt"):
    with open(path, "r") as f:
        me = int(f.readline().strip())
        prev = [list(map(int, list(f.readline().strip()))) for _ in range(N)]
        cur  = [list(map(int, list(f.readline().strip()))) for _ in range(N)]
    return me, prev, cur

def write_output(move, path="output.txt"):
    with open(path, "w") as f:
        if move is None:
            f.write("PASS\n")
        else:
            i, j = move
            f.write(f"{i},{j}\n")

# ---------- Board helpers ----------
def neighbors(i, j):
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < N and 0 <= nj < N:
            yield ni, nj

def collect_group(board, i, j):
    color = board[i][j]
    stack = [(i,j)]
    seen = {(i,j)}
    while stack:
        x, y = stack.pop()
        for nx, ny in neighbors(x, y):
            if board[nx][ny] == color and (nx,ny) not in seen:
                seen.add((nx,ny))
                stack.append((nx,ny))
    return seen

def group_liberties(board, group):
    libs = set()
    for x, y in group:
        for nx, ny in neighbors(x, y):
            if board[nx][ny] == EMPTY:
                libs.add((nx,ny))
    return libs

def remove_group(board, group):
    for x, y in group:
        board[x][y] = EMPTY

def simulate_play(board, move, color):
    """Return (new_board, captured_count) or (None, 0) if illegal (incl. suicide w/o capture)."""
    if move is None:  # PASS
        return deepcopy(board), 0
    i, j = move
    if board[i][j] != EMPTY:
        return None, 0
    newb = [row[:] for row in board]
    newb[i][j] = color

    opp = 3 - color
    captured = 0
    to_remove = []
    seen = set()
    for ni, nj in neighbors(i, j):
        if newb[ni][nj] == opp and (ni,nj) not in seen:
            g = collect_group(newb, ni, nj)
            seen |= g
            if len(group_liberties(newb, g)) == 0:
                to_remove.append(g)
    for g in to_remove:
        captured += len(g)
        remove_group(newb, g)

    # no-suicide (unless captured someone)
    my_group = collect_group(newb, i, j)
    if len(group_liberties(newb, my_group)) == 0 and captured == 0:
        return None, 0

    return newb, captured

def boards_equal(a, b):
    for i in range(N):
        if a[i] != b[i]:
            return False
    return True

def legal_moves(prev_board, cur_board, color):
    moves = []
    for i in range(N):
        for j in range(N):
            if cur_board[i][j] != EMPTY:
                continue
            nb, _ = simulate_play(cur_board, (i,j), color)
            if nb is None:
                continue
            # KO: resulting board cannot equal previous board
            if prev_board is not None and boards_equal(nb, prev_board):
                continue
            moves.append((i,j))
    moves.append(None)  # PASS always legal
    return moves

# ---------- Evaluation ----------
def evaluate(board, color, move_count):
    opp = 3 - color
    my_stones = sum(cell == color for row in board for cell in row)
    opp_stones = sum(cell == opp for row in board for cell in row)
    base = my_stones - opp_stones + (KOMI if color == WHITE else 0.0)

    def all_groups(c):
        seen = set()
        groups = []
        for i in range(N):
            for j in range(N):
                if board[i][j] == c and (i,j) not in seen:
                    g = collect_group(board, i, j)
                    seen |= g
                    groups.append(g)
        return groups

    w_lib = 0.5
    my_lib = sum(len(group_liberties(board, g)) for g in all_groups(color))
    opp_lib = sum(len(group_liberties(board, g)) for g in all_groups(opp))

    pos_bias = 0.0
    fade = max(0.0, 1.0 - move_count/12.0)   # fades out by ~12 stones on board
    if fade > 0.0:
        for i in range(N):
            for j in range(N):
                if board[i][j] == color:
                    if (i in (0, N-1)) and (j in (0, N-1)):
                        pos_bias += 0.15 * fade
                    elif i in (0, N-1) or j in (0, N-1):
                        pos_bias += 0.05 * fade


    return base + w_lib * (my_lib - opp_lib) + pos_bias

def ordering_score(board, move, color):
    if move is None: return -1.0  # prefer real moves first
    nb, captured = simulate_play(board, move, color)
    if nb is None: return -1e9
    score = 5.0 * captured
    opp = 3 - color
    for ni, nj in neighbors(*move):
        if nb[ni][nj] == opp:
            g = collect_group(nb, ni, nj)
            if len(group_liberties(nb, g)) == 1:
                score += 1.0
    g = collect_group(nb, move[0], move[1])
    if len(group_liberties(nb, g)) == 1:
        score -= 1.5
    i, j = move
    if (i in (0, N-1)) and (j in (0, N-1)): score += 0.1
    elif i in (0, N-1) or j in (0, N-1):   score += 0.05
    return score

# ---------- Quiescence (captures/atari only) ----------
def capture_moves(prev_board, cur_board, color):
    caps = []
    for m in legal_moves(prev_board, cur_board, color):
        if m is None: continue
        nb, captured = simulate_play(cur_board, m, color)
        if nb is None: continue
        if captured > 0:
            caps.append((m, nb))
        else:
            # also allow cheap checks: moves that put an enemy group into atari
            opp = 3 - color
            atari = False
            i, j = m
            for ni, nj in neighbors(i, j):
                if nb[ni][nj] == opp:
                    g = collect_group(nb, ni, nj)
                    if len(group_liberties(nb, g)) == 1:
                        atari = True; break
            if atari:
                caps.append((m, nb))
    return caps

# ---------- Searcher ----------
class Searcher:
    def __init__(self, me, prev_board, cur_board):
        self.me = me
        self.prev = prev_board
        self.start_board = cur_board
        self.start_time = time.process_time()
        self.time_limit = 9.7  # seconds CPU time
        self.best_move = None
        self.move_count = sum(cell != EMPTY for row in cur_board for cell in row)

        # Enhancements
        self.TT = {}                 # key -> (depth, flag, value, best_move)
        self.FLAG_EXACT = 0; self.FLAG_LOWER = 1; self.FLAG_UPPER = 2
        self.max_depth = 5
        if self.me == BLACK and self.move_count < 2:
            self.max_depth = min(self.max_depth + 1, 7)  # e.g., 6 if default is 5
            self.time_limit += 0.8 
        self.killers = [[None, None] for _ in range(64)]  # two killers per depth
        self.history = {}            # move_id -> score
        self.pv = {}                 # key-> pv move
        self.nodes = 0

    # --- helpers for TT/pv/ordering ---
    def key(self, prev_b, cur_b, to_play):
        return (tuple(map(tuple, prev_b)) if prev_b is not None else None,
                tuple(map(tuple, cur_b)), to_play)


    def move_id(self, move):
        if move is None: return N*N  # PASS id
        return move[0]*N + move[1]

    def time_up(self):
        return (time.process_time() - self.start_time) >= self.time_limit

    # --- Iterative deepening with aspiration windows ---
    def search(self):
        depth = 1
        last_score = 0.0
        last_best = None
        while depth <= self.max_depth:
            if self.time_up(): break
            # aspiration window around last_score
            window = (0.5 + 0.1*depth) if depth > 1 else 1e6
            alpha = last_score - window
            beta  = last_score + window

            score, bm = self.alpha_beta_root(depth, alpha, beta)

            if (score <= alpha or score >= beta) and not self.time_up():
                if self.time_limit - (time.process_time() - self.start_time) > 0.3:
                    score, bm = self.alpha_beta_root(depth, -1e9, 1e9)


            if bm is not None:
                last_best = bm
                last_score = score
            depth += 1
        return last_best if last_best is not None else self.fallback_move()

    # --- Root search ---
    def alpha_beta_root(self, depth, alpha, beta):
        color = self.me
        moves = legal_moves(self.prev, self.start_board, color)
        if self.move_count < 4 and None in moves:
            moves.remove(None)
            moves.append(None)

        if (self.me == BLACK
            and self.prev is not None
            and all(all(c==EMPTY for c in row) for row in self.prev)
            and all(all(c==EMPTY for c in row) for row in self.start_board)):
            for c in [(0,0),(0,4),(4,0),(4,4)]:
                if c in moves:
                    moves.remove(c)
                    moves.insert(0, c)
                    break

        # PV first (if exists)
        pv_move = self.pv.get(self.key(self.prev, self.start_board, color))
        if pv_move in moves:
            moves.remove(pv_move)
            moves.insert(0, pv_move)

        # order moves by (killer/history/ordering)
        def score_root(m):
            sc = ordering_score(self.start_board, m, color)
            mid = self.move_id(m)
            sc += 0.01 * self.history.get(mid, 0)
            return sc
        moves.sort(key=score_root, reverse=True)

        best_val = -1e9
        best = None
        for m in moves:
            if self.time_up(): break
            nb, _ = simulate_play(self.start_board, m, color)
            if nb is None: continue
            val = self.alpha_beta(self.prev, self.start_board, nb, 3 - color, depth - 1, alpha, beta)
            if val > best_val:
                best_val, best = val, m
                if val > alpha:
                    alpha = val
                    # store PV at root
                    self.pv[self.key(self.prev, self.start_board, color)] = m
            if alpha >= beta:
                break
        self.best_move = best
        return best_val, best

       # --- Quiescence search (captures & atari only) ---
    def quiescence(self, prev_board, cur_board, color, alpha, beta, qdepth=10, seen=None):
        """
        Stand-pat with capture/atari extensions only.
        - qdepth caps recursion in tactical storms.
        - seen prevents rare loops beyond KO (simple superko-ish guard).
        """
        if qdepth <= 0 or self.time_up():
            return evaluate(cur_board, self.me, self.move_count)

        # repetition guard (board + side)
        key_seen = (tuple(map(tuple, cur_board)), color)
        if seen is None:
            seen = set()
        else:
            if key_seen in seen:
                return evaluate(cur_board, self.me, self.move_count)
        seen.add(key_seen)

        stand_pat = evaluate(cur_board, self.me, self.move_count)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        caps = capture_moves(prev_board, cur_board, color)
        if not caps:
            return stand_pat
        # order tactical replies by capture/atari goodness
        caps.sort(key=lambda t: ordering_score(cur_board, t[0], color), reverse=True)

        for m, nb in caps:
            if self.time_up(): break
            v = -self.quiescence(cur_board, nb, 3 - color, -beta, -alpha, qdepth - 1, seen)
            if v >= beta:
                return v
            if v > alpha:
                alpha = v
        return alpha


    # --- Main alpha-beta ---
    def alpha_beta(self, prev_board, cur_board, next_board, color, depth, alpha, beta):
        alpha_orig, beta_orig = alpha, beta
        if self.time_up():
            return evaluate(next_board, self.me, self.move_count)

        self.nodes += 1
        to_play = color
        # TT probe
        k = self.key(prev_board, next_board, to_play)
        hit = self.TT.get(k)
        if hit is not None:
            d_stored, flag, val, mv = hit
            if d_stored >= depth:
                if flag == self.FLAG_EXACT:
                    return val
                elif flag == self.FLAG_LOWER and val > alpha:
                    alpha = val
                elif flag == self.FLAG_UPPER and val < beta:
                    beta = val
                if alpha >= beta:
                    return val
            # try PV move first if we have one
            if mv is not None:
                self.pv[k] = mv

        # depth / leaf
        if depth == 0:
            # quiescence to resolve cheap tactics
            return self.quiescence(prev_board, next_board, to_play, alpha, beta, qdepth=10)

        # Generate legal moves from (prev=cur_board, cur=next_board)
        moves = legal_moves(cur_board, next_board, to_play)

        # PV move first
        pv_move = self.pv.get(k)
        if pv_move in moves:
            moves.remove(pv_move)
            moves.insert(0, pv_move)

        # Killer moves next
        killers = self.killers[depth]
        for km in killers:
            if km is not None and km in moves:
                moves.remove(km)
                moves.insert(0, km)

        # History and static ordering
        def score_child(m):
            sc = ordering_score(next_board, m, to_play)
            mid = self.move_id(m)
            sc += 0.01 * self.history.get(mid, 0)
            # light bonus for killer appearance even if not at top
            if killers[0] == m: sc += 0.2
            if killers[1] == m: sc += 0.1
            return sc
        moves.sort(key=score_child, reverse=True)

        best_move = None

        if to_play == self.me:
            val = -1e9
            for m in moves:
                nb, _ = simulate_play(next_board, m, to_play)
                if nb is None: continue
                v = self.alpha_beta(cur_board, next_board, nb, 3 - to_play, depth - 1, alpha, beta)
                if v > val:
                    val = v; best_move = m
                if v > alpha:
                    alpha = v
                    # store PV edge
                    self.pv[k] = m
                if alpha >= beta:
                    # update killer & history
                    mid = self.move_id(m)
                    if killers[0] != m:
                        killers[1] = killers[0]
                        killers[0] = m
                    self.history[mid] = self.history.get(mid, 0) + depth*depth
                    break
            # store TT using original window
            if val >= beta_orig:
                flag = self.FLAG_LOWER       # fail-high
            elif val <= alpha_orig:
                flag = self.FLAG_UPPER       # fail-low
            else:
                flag = self.FLAG_EXACT
            self.TT[k] = (depth, flag, val, best_move)
            return val
        else:
            val = 1e9
            for m in moves:
                nb, _ = simulate_play(next_board, m, to_play)
                if nb is None: continue
                v = self.alpha_beta(cur_board, next_board, nb, 3 - to_play, depth - 1, alpha, beta)
                if v < val:
                    val = v; best_move = m
                if v < beta:
                    beta = v
                    self.pv[k] = m
                if alpha >= beta:
                    mid = self.move_id(m)
                    if killers[0] != m:
                        killers[1] = killers[0]
                        killers[0] = m
                    self.history[mid] = self.history.get(mid, 0) + depth*depth
                    break
            # store TT using original window
            if val <= alpha_orig:
                flag = self.FLAG_UPPER       # fail-low
            elif val >= beta_orig:
                flag = self.FLAG_LOWER       # fail-high
            else:
                flag = self.FLAG_EXACT
            self.TT[k] = (depth, flag, val, best_move)
            return val



    def fallback_move(self):
        cand = legal_moves(self.prev, self.start_board, self.me)
        cand = [m for m in cand if m is not None]
        if not cand:
            return None
        cand.sort(key=lambda m: ordering_score(self.start_board, m, self.me), reverse=True)
        return cand[0] if cand else None

# ---------- Entrypoints ----------
def main():
    me, prev, cur = read_input()
    searcher = Searcher(me, prev, cur)
    move = searcher.search()
    write_output(move)

def pick_move(prev_board, cur_board, me):
    searcher = Searcher(me, prev_board, cur_board)
    return searcher.search()

if __name__ == "__main__":
    main()
