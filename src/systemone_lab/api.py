from pathlib import Path
import os, torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .model import SystemOneModel
from .training import load_checkpoint, pick_device
from .tokenizer import LabTokenizer
from .formatting import pack_request, branch_attention_mask
import torch.nn.functional as F

class Request(BaseModel):
    model: str | None = "local-s1"
    state: object
    questions: dict[str,dict]

CKPT=os.environ.get("S1_CHECKPOINT","checkpoints/spatial.pt")
device=pick_device(); model=None; tok=None
app=FastAPI(title="SystemOne Lab")

def ensure_loaded():
    global model,tok
    if model is None:
        if not Path(CKPT).exists(): raise HTTPException(503,f"Checkpoint not found: {CKPT}")
        model,ck=load_checkpoint(CKPT,SystemOneModel,device); model.to(device).eval(); tok=LabTokenizer(ck["tokenizer"])

@app.get("/health")
def health(): return {"ok":True,"checkpoint":CKPT,"device":str(device)}

@app.post("/v1/systemone")
def systemone(req: Request):
    ensure_loaded(); packed=pack_request(tok,req.state,req.questions,device=device,isolate_options=model.isolated_options); mask=model.attention_mask(packed)
    with torch.no_grad():
        h=model.hidden(packed.input_ids[None,:],packed.position_ids[None,:],mask)
        answers={}
        for layout in packed.layouts:
            logits=model.decision_logits(h,layout.decide_position,layout.option_end_positions,
                                         query_entity_token_ids=layout.query_entity_token_ids,
                                         query_entity_state_positions=layout.query_entity_state_positions); p=F.softmax(logits.float(),-1).cpu().tolist(); probs=dict(zip(layout.option_keys,p)); best=max(range(len(p)),key=p.__getitem__)
            if layout.qtype=="noul": answers[layout.key]={"noul":probs.get("true",p[1])}
            elif layout.qtype=="choice":
                K=len(p); pm=max(p); conf=(pm-1/K)/(1-1/K) if K>1 else 1.0
                answers[layout.key]={"choice":layout.option_keys[best],"probabilities":probs,"confidence":conf}
            else:
                expected=sum(i*v for i,v in enumerate(p)); K=len(p); pm=max(p); conf=(pm-1/K)/(1-1/K) if K>1 else 1.0
                answers[layout.key]={"score":expected,"probabilities":probs,"confidence":conf}
    return {"model":"local-s1","answers":answers}

UI=Path(__file__).resolve().parents[2]/"ui"/"index.html"
GAME=UI.parent/"game.html"
CHESS=UI.parent/"chess.html"
try:
    from .chess_api import router as chess_router
    app.include_router(chess_router)
except ImportError:  # python-chess is optional; the spatial API works without it
    chess_router=None

@app.get("/chess")
def chess_page():
    if chess_router is None or not CHESS.exists(): raise HTTPException(404,"chess support not installed")
    return FileResponse(CHESS)

@app.get("/game")
def game():
    if not GAME.exists(): raise HTTPException(404,"game page not found")
    return FileResponse(GAME)

@app.get("/")
def root():
    return FileResponse(UI) if UI.exists() else {"message":"POST /v1/systemone"}
