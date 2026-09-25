// Render chess requests with the browser port (chess.js + chessfmt.js) for comparison with Python.
import { readFileSync } from "node:fs";
import { Chess } from "../../site/vendor/chess.js";
import { request } from "../../site/js/chessfmt.js";
const fens = JSON.parse(readFileSync(process.argv[2], "utf8"));
process.stdout.write(JSON.stringify(fens.map(f => request(new Chess(f)))));
