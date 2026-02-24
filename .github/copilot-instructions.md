# NeuroPlan AI Coding Agent Instructions

## Project Overview
NeuroPlan is a dual-mode study planner combining AI task scheduling with EEG brainwave analysis. 

**Architecture**: Hybrid Next.js + Python system with two main modes:
- **Light Mode** (`/light`): React-based task management with AI planning
- **EEG Mode** (`/eeg`): Streamlit-based brainwave analysis with Muse device integration

## Critical Type System Patterns

**Dual Type Systems**: The project uses two separate Task type systems:
- `src/types.ts`: Legacy planner types with `Task.subject`, `Task.difficulty`, `Task.due` 
- `src/app/light/types.ts`: New UI types with `Task.category`, `Task.estimatedTime`, `Task.dueDate`

Always check which type system a file uses before making changes. API routes in `/api/tasks/` use the light mode types.

**State Management**: Zustand store (`src/state/store.ts`) handles global state. Uses functional updates and nanoid for IDs.

## Key Architectural Decisions

**Planning Algorithm**: `src/lib/planner.ts` implements Pomodoro-based scheduling:
- 25-minute blocks, daily time windows (14:00-16:00, 19:00-21:00)
- Task prioritization: due date first, then difficulty
- Generates `todayTop3` and `weekSlots` arrays

**EEG Integration**: Python-based ML pipeline:
- WebSocket server (`muse_ws.py`) for real-time EEG data
- Model persistence in `model_store/` directory
- Streamlit app embedded via iframe with connection status checking

**API Patterns**: Next.js API routes use:
- `export const dynamic = "force-dynamic"` for real-time data
- In-memory data storage (`src/app/api/tasks/data.ts`)
- RESTful endpoints: `/api/tasks/getAll`, `/api/tasks/add`, `/api/plan`

## Development Workflows

**Frontend Development**:
```bash
npm run dev --turbopack  # Uses Turbopack for faster builds
```

**EEG Mode Development**: 
- Streamlit runs separately on port 8501
- Connection checked via no-cors fetch from Next.js
- Python imports use complex path resolution for module compatibility

**Model Training**: ML models stored as `.model` + `.json` pairs in `model_store/`

## UI Component Patterns

**Design System**: Tailwind CSS with consistent patterns:
- Cards: `rounded-2xl border bg-white p-4 shadow-sm`
- Hover effects: `hover:-translate-y-1 hover:shadow-md`
- Status indicators: Colored dots for connection states

**Client Components**: Most components use `"use client"` directive. Calendar component uses dynamic import with `ssr: false`.

**Task Categories**: UI displays "Deep/Normal/Quick" based on estimated time thresholds (≤10min=quick, <45min=normal, ≥45min=deep).

## Integration Points

**WebSocket Communication**: EEG data flows through WebSocket server with envelope format:
```json
{"source": "muse-ws", "ts": timestamp, "payload": {...}}
```

**Model Loading**: Thread-safe singleton pattern in `eeg_memory_readiness.py` with `load_once()` function.

**Cross-Mode Navigation**: Home page (`/`) provides mode selection with visual status indicators.

## Project-Specific Conventions

- Use `nanoid()` for generating unique IDs
- Date strings in `YYYY-MM-DD` format, times as Date objects or ISO strings  
- Error handling: Try/catch with user-friendly messages, console.error for debugging
- File imports: Use `@/` aliases for src directory, relative imports within same directory
- Python modules: Complex path manipulation for cross-compatibility between execution contexts

## Dependencies & External Systems

**Key NPM packages**: `zustand` (state), `dayjs` (dates), `@fullcalendar/react` (calendar), `@dnd-kit/*` (drag-drop)

**Python Stack**: `streamlit`, `numpy`, `sklearn`, `pylsl` (optional), `websockets`

**Environment Variables**: 
- `NEXT_PUBLIC_EEG_URL` for Streamlit integration
- Development uses localhost:3000 (Next.js) + localhost:8501 (Streamlit)