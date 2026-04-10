"""FastAPI app exposing reset/step/state APIs for OpenEnv validation."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from openenv_selfdriving.environment import SelfDrivingOpenEnv
from openenv_selfdriving.models import GradeReport, SelfDrivingAction, SelfDrivingObservation, SelfDrivingState, StepOutput, TaskSpec


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_open_road")
    seed: int | None = None


env = SelfDrivingOpenEnv(seed=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    env.reset(task_id="easy_open_road", seed=42)
    yield


app = FastAPI(title="Autonomous Traffic Agent System", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Autonomous Traffic Agent System</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #1e1e1e; color: #fff; }
            canvas { background: #333; margin-top: 20px; border: 2px solid #555; }
            button { padding: 10px 20px; margin: 10px; font-size: 16px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Autonomous Traffic Agent System Dashboard</h1>
        <p>This is the monitoring view for the autonomous traffic optimizer.</p>
        <button onclick="reset()">Reset Episode</button>
        <button onclick="startAuto()">Auto Step</button>
        <br>
        <canvas id="simCanvas" width="800" height="200"></canvas>
        <p id="status"></p>

        <script>
            let ctx = document.getElementById("simCanvas").getContext("2d");
            let autoInterval = null;

            async function fetchState() {
                let res = await fetch("/state");
                let data = await res.json();
                draw(data);
                if (data.collisions > 0 || data.reached_goal) clearInterval(autoInterval);
            }

            async function reset() {
                clearInterval(autoInterval);
                await fetch("/reset", { method: "POST", headers: {"Content-Type": "application/json"}, body: "{}" });
                fetchState();
            }

            async function step() {
                // Request a step from the autonomous simulation
                let state_res = await fetch("/state");
                let state = await state_res.json();

                let action = "maintain";
                // simple heuristic for baseline demo viewing
                for (let obs of state.obstacles) {
                    if (obs.lane === state.ego.lane && obs.position_m > state.ego.position_m && (obs.position_m - state.ego.position_m) < 40) {
                        action = (state.ego.lane === 0) ? "lane_right" : "lane_left";
                    }
                }

                let res = await fetch("/step", { 
                    method: "POST", 
                    headers: {"Content-Type": "application/json"}, 
                    body: JSON.stringify({ action: action }) 
                });
                fetchState();
            }

            function startAuto() {
                clearInterval(autoInterval);
                autoInterval = setInterval(step, 500);
            }

            function draw(state) {
                ctx.clearRect(0, 0, 800, 200);
                
                // Draw lanes
                ctx.strokeStyle = "#888";
                ctx.setLineDash([20, 15]);
                for (let i=1; i<3; i++) {
                    ctx.beginPath(); ctx.moveTo(0, i*66); ctx.lineTo(800, i*66); ctx.stroke();
                }
                
                // Scale x
                let scaleX = 800 / Math.max(100, state.goal_position_m);

                // Draw obstacles
                ctx.fillStyle = "red";
                state.obstacles.forEach(obs => {
                    ctx.fillRect(obs.position_m * scaleX, obs.lane * 66 + 10, 30, 46);
                });

                // Draw ego
                ctx.fillStyle = "green";
                ctx.fillRect(state.ego.position_m * scaleX, state.ego.lane * 66 + 10, 30, 46);

                let info = `Step: ${state.step_count} | Collisions: ${state.collisions} | Ego Pos: ${state.ego.position_m.toFixed(1)}m`;
                document.getElementById("status").innerText = info;
            }

            fetchState();
        </script>
    </body>
    </html>
    """



@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", response_model=list[TaskSpec])
async def tasks() -> list[TaskSpec]:
    return env.list_tasks()


@app.post("/reset", response_model=SelfDrivingObservation)
async def reset(req: ResetRequest | None = None) -> SelfDrivingObservation:
    if req is None:
        req = ResetRequest()
    return env.reset(task_id=req.task_id, seed=req.seed)

@app.post("/step", response_model=StepOutput)
async def step(action: SelfDrivingAction) -> StepOutput:
    return env.step(action)


@app.get("/state", response_model=SelfDrivingState)
async def state() -> SelfDrivingState:
    return env.state()


@app.get("/grade", response_model=GradeReport)
async def grade() -> GradeReport:
    return env.grade_current_episode()
