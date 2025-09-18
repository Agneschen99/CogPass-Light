// src/state/store.ts
import { create } from "zustand";
import { nanoid } from "nanoid";
import type { Task, Slot, CheckIn } from "@/types";
import { number } from "zod";

// If your Slot type doesn't have "done", you can keep it optional.
// Example Slot extension used only in UI toggles:
export type UISlot = Slot & { done?: boolean };

type Plan = {
  todayTop3: UISlot[];
  weekSlots: UISlot[];
};

type Store = {
  // Data
  tasks: Task[];
  pomodoros: number;
  todayTop3: UISlot[];
  weekSlots: UISlot[];
  checkIns: CheckIn[];

  // Task ops
  addTask: (t: Omit<Task, "id">) => void;
  removeTask: (id: string) => void;
  clearTasks: () => void;

  // Plan ops
  setPlan: (plan: Plan) => void;
  resetPlan: () => void;
  toggleSlotDone: (slotId: string, force?: boolean) => void;

  // Check-ins
  addCheckIn: (c: CheckIn) => void;
};

export const useStore = create<Store>((set) => ({
  // ---------- SAFE DEFAULTS ----------
  tasks: [],                // never undefined
  pomodoros: 0,
  todayTop3: [],            // never undefined
  weekSlots: [],            // never undefined
  checkIns: [],             // never undefined

  // ---------- TASKS ----------
addTask: (t) =>
  set((s) => ({
    tasks: [
      ...s.tasks,
      {
        id: nanoid(),
        title: t.title,
        subject: t.subject,
        difficulty: t.difficulty,
        due: t.due,
        minutes: t.minutes,
      },
    ],
  })), 
  
  removeTask: (id: string) =>
    set((s) => ({ tasks: s.tasks.filter((task) => task.id !== id) })),

  clearTasks: () => set({ tasks: [] }),

  // ---------- PLAN ----------
  setPlan: (plan) =>
    set(() => ({
      todayTop3: plan.todayTop3 ?? [],
      weekSlots: plan.weekSlots ?? [],
    })),

  resetPlan: () =>
    set(() => ({
      todayTop3: [],
      weekSlots: [],
    })),

  toggleSlotDone: (slotId, force) =>
    set((s) => ({
      weekSlots: s.weekSlots.map((sl) =>
        sl.id === slotId ? { ...sl, done: force ?? !sl.done } : sl
      ),
      todayTop3: s.todayTop3.map((sl) =>
        sl.id === slotId ? { ...sl, done: force ?? !sl.done } : sl
      ),
    })),

  // ---------- CHECK-INS ----------
  addCheckIn: (c) => set((s) => ({ checkIns: [...s.checkIns, c] })),
}));
