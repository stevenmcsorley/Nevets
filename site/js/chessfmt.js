// Browser port of src/systemone_lab/chess_format.py on top of chess.js. tests/test_js_parity checks
// that it renders states and moves byte-for-byte like the Python version the model was trained on.
const ORDER = "KQRBNP";
const FILES = "abcdefgh";
const sqIndex = sq => (Number(sq[1]) - 1) * 8 + FILES.indexOf(sq[0]);  // python-chess square order

export function stateText(game) {
  const side = game.turn() === "w" ? "White" : "Black";
  const parts = [`${side} to move.`];
  const cells = game.board().flat().filter(Boolean);
  for (const [colour, name] of [["w", "White"], ["b", "Black"]]) {
    const pieces = [];
    for (const sym of ORDER) {
      cells.filter(c => c.color === colour && c.type === sym.toLowerCase())
           .sort((a, b) => sqIndex(a.square) - sqIndex(b.square))
           .forEach(c => pieces.push(sym + c.square));
    }
    parts.push(`${name}: ${pieces.join(" ")}.`);
  }
  const castling = game.fen().split(" ")[2];
  parts.push(`Castling: ${castling !== "-" ? castling : "none"}.`);
  // Like python-chess has_legal_en_passant(): only report the square if an en-passant capture is legal.
  const ep = game.moves({ verbose: true }).find(m => m.isEnPassant());
  parts.push(`En passant: ${ep ? ep.to : "none"}.`);
  if (game.inCheck()) parts.push("Check.");
  return parts.join(" ");
}

export const uci = m => m.from + m.to + (m.promotion || "");

export function moveText(m) {
  let text = m.piece.toUpperCase() + m.from + (m.isCapture() || m.isEnPassant() ? "x" : "-") + m.to;
  if (m.promotion) text += "=" + m.promotion.toUpperCase();
  if (m.isKingsideCastle() || m.isQueensideCastle()) text += " castles";
  return text;
}

export function request(game) {
  const side = game.turn() === "w" ? "White" : "Black";
  const moves = game.moves({ verbose: true }).sort((a, b) => (uci(a) < uci(b) ? -1 : uci(a) > uci(b) ? 1 : 0));
  return { state: stateText(game), questions: { move: { type: "choice", instructions: `Which move is best for ${side}?`,
           criteria: Object.fromEntries(moves.map(m => [uci(m), moveText(m)])) } } };
}
