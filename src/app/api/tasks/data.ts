import { randomUUID } from "crypto";
import type { Task } from "@/app/light/types";

function classify(estimatedTime: number): Task["category"] {
  if (estimatedTime <= 10) return "quick";
  if (estimatedTime < 45) return "normal";
  return "deep";
}

let tasks: Task[] = [
  {
    id: randomUUID(),
    title: "Review math chapter 3",
    estimatedTime: 50,
    category: "deep",
  },
  {
    id: randomUUID(),
    title: "Summarize history notes",
    estimatedTime: 30,
    category: "normal",
  },
  {
    id: randomUUID(),
    title: "English vocab drill",
    estimatedTime: 20,
    category: "normal",
  },
  {
    id: randomUUID(),
    title: "Flashcards sprint",
    estimatedTime: 8,
    category: "quick",
  },
];

let top3Ids: string[] = [];

export function getTasks(): Task[] {
  return tasks;
}

export function addTask(input: { title: string; estimatedTime: number; dueDate?: string }): Task {
  const task: Task = {
    id: randomUUID(),
    title: input.title,
    estimatedTime: input.estimatedTime,
    dueDate: input.dueDate,
    category: classify(input.estimatedTime),
  };
  tasks = [...tasks, task];
  return task;
}

export function setTop3(taskId: string): Task[] {
  if (!tasks.find((t) => t.id === taskId)) return getTop3();
  top3Ids = [taskId, ...top3Ids.filter((id) => id !== taskId)].slice(0, 3);
  return getTop3();
}

export function getTop3(): Task[] {
  return top3Ids
    .map((id) => tasks.find((t) => t.id === id))
    .filter(Boolean) as Task[];
}

// ✅ 新增：从 top3 移除某个任务
export function removeFromTop3(taskId: string): Task[] {
  top3Ids = top3Ids.filter((id) => id !== taskId);
  return getTop3();
}

// ✅ 新增：删除任务（同时把它从 top3Ids 移除）
export function deleteTask(taskId: string): { tasks: Task[]; top3: Task[] } {
  // 1) 删 tasks
  tasks = tasks.filter((t) => t.id !== taskId);

  // 2) top3Ids 里也删掉
  top3Ids = top3Ids.filter((id) => id !== taskId);

  return { tasks: getTasks(), top3: getTop3() };
}
