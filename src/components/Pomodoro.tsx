"use client";
import { useEffect, useRef, useState } from "react";

type Props = { onBreak?: (ts: string) => void };

export default function Pomodoro({ onBreak }: Props) {
  const [seconds, setSeconds] = useState(25 * 60);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<'work' | 'break'>('work');
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!running) return;
    timerRef.current = window.setInterval(() => setSeconds((s) => s - 1), 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [running]);

  useEffect(() => {
    if (seconds === 0) {
      if (phase === 'work') {
        setPhase('break');
        setSeconds(5 * 60);
        setRunning(false);
        onBreak?.(new Date().toISOString());
      } else {
        setPhase('work');
        setSeconds(25 * 60);
        setRunning(false);
      }
    }
  }, [seconds, phase, onBreak]);

  const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
  const ss = String(seconds % 60).padStart(2, '0');

  return (
    <div className="flex items-center justify-between gap-6">
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-5 py-4">
        <div className="text-4xl font-semibold tabular-nums">{mm}:{ss}</div>
        <div className="mt-1 text-xs text-slate-500">Phase: {phase}</div>
      </div>
      <div className="flex gap-2">
        <button onClick={() => setRunning(true)} className="rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500">Start</button>
        <button onClick={() => setRunning(false)} className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50">Pause</button>
        <button
          onClick={() => { setRunning(false); setPhase('work'); setSeconds(25 * 60); }}
          className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium hover:bg-slate-50"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
