// src/app/page.tsx
"use client";

import type { FormEvent } from "react";
import React, { useMemo, useState } from "react";
import dayjs from "dayjs";
import EEGStatus from '@/components/EEGStatus';
// ...
<EEGStatus />

// ---- FullCalendar（周视图）
import dynamic from "next/dynamic";
const Calendar = dynamic(() => import("@/components/Calendar"), { ssr: false });


import { useStore } from "@/state/store";
import { exportWeekToICS } from "@/lib/ics";
import { generatePlan } from "@/lib/planner";
import { parseChatInput } from "@/lib/nlp";
import type { Task, Slot } from "@/types";

export default function Home() {
  // ---- state from store
  const { tasks, plan, addTask, setPlan, toggleDone, clearAll } = useStore();
  function onGenerate() {
  const planData = generatePlan(tasks);
  setPlan(planData);
}

async function onGenerateServer() {
  setAiLoading(true);
  try {
    const resp = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tasks }),
    });
    const data = await resp.json();
    setPlan(data);
  } finally {
    setAiLoading(false);
  }
}
// ---- convert weekSlots -> FullCalendar events ----
const slotEvents = plan?.weekSlots?.map((slot: any, i: number) => ({
  id: slot.id ?? `slot-${i}`,
  title: slot.title || slot.subject || "Task",
  start: slot.start,
  end: slot.end,
  extendedProps: { kind: "slot" },
})) ?? [];

const taskEvents = tasks
  ?.filter((t: any) => !!t.due)
  .map((t: any, i: number) => ({
    id: t.id ?? `task-${i}`,
    title: t.title || t.subject || "Task",
    start: dayjs(t.due).toISOString(),
    end: dayjs(t.due).add(t.minutes ?? 25, "minute").toISOString(),
    extendedProps: { kind: "task" },
  })) ?? [];

// ---- 合并给 FullCalendar ----
const fcEvents = [...slotEvents, ...taskEvents];
  // ---- local form states
  const [title, setTitle] = useState("");
  const [subject, setSubject] = useState("");
  const [difficulty, setDifficulty] = useState<number>(3);
  const [minutes, setMinutes] = useState<number>(120);
  const [due, setDue] = useState<string>("");
const [aiLoading, setAiLoading] = useState(false);
  // ---- chat box state
  const [chat, setChat] = useState<string>("");

  // derived counts
  const scheduledCount = useMemo(() => {
    return plan?.weekSlots?.length ?? 0;
  }, [plan]);

  // =========================
  // 👇 新增：调用 /api/plan（你的后端/模型）
  // =========================
  async function callPlanner() {
    try {
      setAiLoading(true);

      const resp = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tasks: useStore.getState().tasks }),
      });

      if (!resp.ok) {
        const msg = await resp.text();
        throw new Error(msg || "Planner API error");
      }

      const data = await resp.json();
      // 期待 data: { todayTop3?: Slot[], weekSlots?: Slot[] }
      const todayTop3: Slot[] = (data?.todayTop3 ?? []).map((s: any) => ({
        ...s,
        start: s.start, // 这里可以保留字符串；渲染时再转 Date
      }));
      const weekSlots: Slot[] = (data?.weekSlots ?? []).map((s: any) => ({
        ...s,
        start: s.start,
      }));

      setPlan({ todayTop3, weekSlots });
    } catch (err: any) {
      console.error(err);
      alert(err?.message ?? "Failed to call planner.");
    } finally {
      setAiLoading(false);
    }
  }

  // ---- handlers
  function onAddTask(e: FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;

    const t: Task = {
      id: crypto.randomUUID(),
      title: title.trim(),
      subject: subject.trim() || "general",
      difficulty: Number(difficulty) || 3,
      minutes: Number(minutes) || 25,
      // 如果你的 Task.due 是 string，就保持 string
      due: due || dayjs().endOf("week").format("YYYY-MM-DD"),
    };
    addTask(t);
    setTitle("");
    setMinutes(120);
  }

  function onGenerate() {
    // 本地算法版本
    setPlan(generatePlan(useStore.getState().tasks, 5));
  }

  function onExportICS() {
    const ics = exportWeekToICS(plan?.weekSlots || []);
    const blob = new Blob([ics], { type: "text/calendar;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.download = "week-plan.ics";
    a.href = url;
    a.click();
    URL.revokeObjectURL(url);
  }

  function onClearPlan() {
    setPlan({ todayTop3: [], weekSlots: [] });
  }

  // ---- chat flow
  function onSubmitChat() {
    const input = chat.trim();
    if (!input) return;

    const { tasks: parsed, action } = parseChatInput(input);

    if (action === "clear") {
      clearAll();
      setPlan({ todayTop3: [], weekSlots: [] });
      setChat("");
      return;
    }

    if (parsed.length) parsed.forEach(addTask);

    if (action === "generate" || parsed.length) {
      const all = useStore.getState().tasks;
      const newPlan = generatePlan(all, 5);
      setPlan(newPlan);
    }

    setChat("");
  }

  // ---- Today top3 helper
  const todayTop3: Slot[] = (plan?.todayTop3 as Slot[]) || [];

  return (
  <div>
    {/* EEG Mode */}
    <div className="flex justify-end px-8 pt-4">
      <a
        href="/eeg"
        className="rounded-md bg-blue-600 text-white px-4 py-2 hover:bg-blue-700 transition"
      >
        🧠 Open EEG Mode
      </a>
    </div>
    <main className="mx-auto max-w-5xl px-6 py-8">
      {/* Header */}
      <header className="mb-6 flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">Plan smarter · study calmer.</p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={onExportICS}
            className="rounded border px-3 py-1 text-sm hover:bg-gray-50"
          >
            Export Week (.ics)
          </button>
          <button
            onClick={onClearPlan}
            className="rounded border px-3 py-1 text-sm hover:bg-gray-50"
          >
            Clear Plan
          </button>
        </div>
      </header>

      {/* Chat input */}
      <section className="mb-8 rounded-xl border p-4">
        <h2 className="mb-3 text-lg font-medium">Quick Add (Chat)</h2>
        <div className="flex flex-col gap-3">
          <textarea
            value={chat}
            onChange={(e) => setChat(e.target.value)}
            placeholder={`Examples:
today buy groceries 30m
next Wed math exam 120m difficulty 4
generate
clear`}
            className="min-h-[100px] w-full rounded border px-3 py-2"
          />
          <div className="flex flex-wrap gap-3">
            <button
              onClick={onSubmitChat}
              className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
            >
              Parse & Add
            </button>

            {/* 本地算法 */}
           <button
  onClick={onGenerate}
  className="rounded border px-4 py-2 text-sm hover:bg-gray-50"
>
  Generate Week Plan
</button>

{/* 服务器端 AI 规划 */}
<button
  onClick={onGenerateServer}
  disabled={aiLoading}
  className="rounded border px-4 py-2 text-sm hover:bg-gray-50 disabled:opacity-60"
>
  {aiLoading ? "AI Generating..." : "AI Generate (server)"}
</button>

<button
  onClick={clearAll}
  className="rounded border px-4 py-2 text-sm hover:bg-gray-50"
>
  Clear Tasks
</button>
          </div>
          <p className="text-xs text-gray-500">
            Tip: You can type multiple commands over time. Use{" "}
            <code>clear</code> to wipe tasks, or <code>generate</code> /{" "}
            <code>AI Generate (server)</code> to make a new plan.
          </p>
        </div>
      </section>

      {/* Classic Add Task form */}
      <section className="mb-8 rounded-xl border p-4">
        <h2 className="mb-3 text-lg font-medium">Add Task</h2>
        <form onSubmit={onAddTask} className="flex flex-wrap items-end gap-3">
          <input
            className="w-[280px] rounded border px-3 py-2"
            placeholder="Title (e.g., Chapter 3 Review)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <input
            className="w-[220px] rounded border px-3 py-2"
            placeholder="Subject (e.g., math / history)"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
          />
          <label className="flex flex-col text-sm">
  Difficulty (1–5)
  <input
    className="w-20 rounded border px-3 py-2 text-center"
    placeholder="e.g., 3"
    value={difficulty}
    onChange={(e) => setDifficulty(Number(e.target.value))}
    type="number"
    min={1}
    max={5}
  />
</label>

<label className="flex flex-col text-sm">
  Estimated Minutes
  <input
    className="w-24 rounded border px-3 py-2 text-center"
    placeholder="e.g., 120"
    value={minutes}
    onChange={(e) => setMinutes(Number(e.target.value))}
    type="number"
    min={10}
    step={5}
  />
</label>
          <input
            className="w-[180px] rounded border px-3 py-2"
            type="date"
            value={due}
            onChange={(e) => setDue(e.target.value)}
            title="Due date"
          />
          <button
            type="submit"
            className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Add Task
          </button>
          <p className="text-sm text-gray-500">
            {tasks.length} task(s) saved · {scheduledCount} scheduled blocks
          </p>
        </form>
      </section>

      {/* Today · Top 3 */}
      <section className="mb-8 rounded-xl border p-4">
        <h2 className="mb-3 text-lg font-medium">Today · Top 3</h2>
        {todayTop3.length === 0 ? (
          <p className="text-sm text-gray-500">
            No items yet. Use <b>Generate Week Plan</b> or{" "}
            <b>AI Generate (server)</b> to create today’s top 3.
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {todayTop3.map((s, idx) => (
              <div key={s.id ?? `${s.title}-${idx}`} className="rounded border p-3">
                <div className="mb-1 text-xs text-gray-500">
                  {dayjs(s.start).format("HH:mm")} · {s.minutes}m
                </div>
                <div className="font-medium">
                  {s.subject} · {s.title}
                </div>
                <label className="mt-2 flex items-center gap-2 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    checked={!!s.done}
                    onChange={() => toggleDone(s.id)}
                  />
                  Mark done
                </label>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Week Plan — FullCalendar + 备选列表 */}
      <section className="mb-16 rounded-xl border p-4">
        <h2 className="mb-4 text-lg font-medium">Week Plan</h2>

        {/* FullCalendar 周视图 */}
        <div className="rounded border">
         <Calendar events={fcEvents} />
        </div>

        {/* 备选：简易列表 */}
        {!plan?.weekSlots?.length ? (
          <p className="mt-4 text-sm text-gray-500">No items scheduled.</p>
        ) : (
          <div className="mt-4 space-y-2">
            {plan.weekSlots.map((s, i) => (
              <div
                key={s.id ?? `${s.title}-${i}`}
                className="flex items-center justify-between rounded border px-3 py-2"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm">
                    <span className="font-medium">{s.subject}</span> · {s.title}
                  </div>
                  <div className="text-xs text-gray-500">
                    {dayjs(s.start).format("ddd · HH:mm")} · {s.minutes}m
                  </div>
                </div>
                <label className="ml-3 flex shrink-0 items-center gap-2 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    checked={!!s.done}
                    onChange={() => toggleDone(s.id)}
                  />
                  to complete
                </label>
              </div>
            ))}
          </div>
        )}
      </section>

      <footer className="pb-10 text-xs text-gray-400">
      
      </footer>
    </main>
  </div>
);
} // ← 这是 export default function Home() 的结束大括号

