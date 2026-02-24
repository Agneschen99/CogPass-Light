export type TaskCategory = "deep" | "normal" | "quick";

export type Task = {
  id: string;
  title: string;
  estimatedTime: number;
  dueDate?: string;
  category: TaskCategory;
};
