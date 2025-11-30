// src/components/TaskList.tsx
"use client";

import { useState } from "react";
import { useStore } from "@/state/store";

export default function TaskList() {
  const { tasks, todayTop3, addTask, toggleDone } = useStore();
  const [newTask, setNewTask] = useState("");

  function handleAddTask() {
    const title = newTask.trim();
    if (!title) return;
    addTask({
      title,
      subject: "general",
      difficulty: 3,
      minutes: 25,
      due: new Date().toISOString().slice(0, 10),
    });
    setNewTask("");
  }

  function moveToTop3(task: any) {
    if (todayTop3.length >= 3) {
      alert("⚠ Top3 已满");
      return;
    }
    if (todayTop3.find((t: any) => t.id === task.id)) return;

    // Convert task into a Slot-like object so it stays compatible with the rest of the app.
    const slot = {
      id: task.id,
      title: task.title,
      subject: task.subject,
      start: task.due ?? new Date().toISOString(),
      minutes: task.minutes,
      done: task.done,
    };
    useStore.setState({ todayTop3: [...todayTop3, slot] });
  }

  return (
    <div
      style={{
        marginTop: 20,
        padding: 16,
        background: "#eef5ff",
        borderRadius: 10,
      }}
    >
      <h2 style={{ fontWeight: 700 }}>📄 All Tasks</h2>

      <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
        <input
          value={newTask}
          onChange={(e) => setNewTask(e.target.value)}
          placeholder="输入任务..."
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #c9d6ff" }}
        />
        <button
          onClick={handleAddTask}
          style={{
            padding: "8px 14px",
            background: "#4f8bff",
            color: "white",
            borderRadius: 6,
            border: "none",
            cursor: "pointer",
          }}
        >
          Add
        </button>
      </div>

      {tasks.length === 0 && <p style={{ marginTop: 10 }}>暂无任务</p>}

      {tasks.map((t: any) => (
        <div
          key={t.id}
          style={{
            background: "white",
            padding: 10,
            marginTop: 8,
            borderRadius: 6,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 10,
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 8, flex: 1 }}>
            <input type="checkbox" checked={!!t.done} onChange={() => toggleDone(t.id)} />
            <span style={{ textDecoration: t.done ? "line-through" : "none" }}>{t.title}</span>
          </label>
          <button
            onClick={() => moveToTop3(t)}
            style={{
              padding: "6px 10px",
              background: "#f3f5ff",
              border: "1px solid #dbe2ff",
              borderRadius: 6,
              cursor: "pointer",
            }}
          >
            ➕ Top3
          </button>
        </div>
      ))}
    </div>
  );
}
