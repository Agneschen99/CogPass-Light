// src/app/light/page.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import dynamic from "next/dynamic";
import Top3List from "./components/Top3List";
import TaskBoard from "./components/TaskBoard";
import type { Task } from "./types";

const Calendar = dynamic(() => import("@/components/Calendar"), { ssr: false });

export default function LightModePage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [top3, setTop3] = useState<Task[]>([]);
  const [title, setTitle] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [estimatedTime, setEstimatedTime] = useState<number>(25);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    refreshAll();
  }, []);

  async function refreshAll() {
    await Promise.all([fetchTasks(), fetchTop3()]);
  }

  async function fetchTasks() {
    const res = await fetch("/api/tasks/getAll");
    const data = (await res.json()) as Task[];
    setTasks(data);
  }

  async function fetchTop3() {
    const res = await fetch("/api/tasks/top3");
    const data = (await res.json()) as Task[];
    setTop3(data);
  }

  async function safeJson(url: string, body: any) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Failed: ${url}`);
    return await res.json();
  }

  async function handleAddTask() {
    const trimmed = title.trim();
    if (!trimmed || estimatedTime <= 0) return;
    setLoading(true);
    try {
      await safeJson("/api/tasks/add", {
        title: trimmed,
        dueDate: dueDate || undefined,
        estimatedTime,
      });
      await refreshAll();
      setTitle("");
      setEstimatedTime(25);
      setDueDate("");
    } catch (err) {
      console.error(err);
      alert("新增任务失败");
    } finally {
      setLoading(false);
    }
  }

  async function handleAddToTop3(taskId: string) {
    setLoading(true);
    try {
      await safeJson("/api/tasks/top3", { id: taskId });
      await refreshAll();
    } catch (err) {
      console.error(err);
      alert("Top3 更新失败");
    } finally {
      setLoading(false);
    }
  }

  async function handleMove(taskId: string, to: Task["category"]) {
    try {
      await safeJson("/api/tasks/updateCategory", { id: taskId, category: to });
      await refreshAll();
    } catch (err) {
      console.error(err);
      alert("更新任务分类失败");
    }
  }

  const calendarEvents = useMemo(() => {
    // 根据任务 id 生成稳定的 slot 索引 (0, 1, 或 2)
    const getStableSlot = (taskId: string): number => {
      // 使用 id 的字符码总和来确定 slot，确保同一个任务总是分配到同一个 slot
      const hash = taskId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
      return hash % 3;
    };

    return (top3 ?? []).map((t) => {
      // 确定日期：有 dueDate 用 dueDate，没有用今天
      const targetDate = t.dueDate ? dayjs(t.dueDate) : dayjs();
      
      // 使用稳定的 slot 索引
      const slotIndex = getStableSlot(t.id);
      const start = targetDate.hour(9 + slotIndex).minute(0).second(0).millisecond(0);
      const end = start.add(t.estimatedTime ?? 25, "minute");
      
      return {
        id: `top3-${t.id}`,
        title: t.title,
        start: start.toISOString(),
        end: end.toISOString(),
        extendedProps: { kind: "slot" },
      };
    });
  }, [top3]);

  return (
    <div className="min-h-screen bg-[#FAFAFA] p-6">
      <header className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-800">☀️ NeuroPlan · Light Mode</h1>
        <div className="text-sm text-gray-500">AI Planning System</div>
      </header>

      <div className="mb-6 grid gap-4 rounded-xl border bg-white p-4 shadow-sm lg:grid-cols-4">
        <div className="lg:col-span-2">
          <label className="text-sm text-gray-600">任务标题</label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g., Finish math worksheet"
            className="mt-1 w-full rounded border px-3 py-2"
          />
        </div>
        <div>
          <label className="text-sm text-gray-600">预计分钟</label>
          <input
            type="number"
            min={5}
            value={estimatedTime}
            onChange={(e) => setEstimatedTime(Number(e.target.value) || 0)}
            className="mt-1 w-full rounded border px-3 py-2"
          />
        </div>
        <div>
          <label className="text-sm text-gray-600">截止日期 (可选)</label>
          <input
            type="date"
            value={dueDate}
            onChange={(e) => setDueDate(e.target.value)}
            className="mt-1 w-full rounded border px-3 py-2"
          />
        </div>
        <div className="lg:col-span-4 flex justify-end">
          <button
            onClick={handleAddTask}
            disabled={loading}
            className="rounded bg-blue-600 px-4 py-2 text-white transition hover:bg-blue-700 disabled:opacity-60"
          >
            {loading ? "Saving..." : "Add Task"}
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-6 lg:flex-row">
        <div className="flex w-full flex-col gap-6 lg:w-2/3">
          <TaskBoard tasks={tasks} onAddToTop3={handleAddToTop3} onMove={handleMove} />
        </div>

        <div className="flex w-full flex-col gap-6 lg:w-1/3">
          <Top3List tasks={top3} />
          <div className="min-h-[360px] w-full rounded-xl border bg-white p-4 shadow-sm">
            <Calendar events={calendarEvents} />
          </div>
        </div>
      </div>
    </div>
  );
}
