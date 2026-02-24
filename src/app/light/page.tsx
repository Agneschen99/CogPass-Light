// src/app/light/page.tsx
"use client";
"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Brain,
  Calendar,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Circle,
  Clock,
  Eye,
  EyeOff,
  Lock,
  LogOut,
  Mail,
  Pause,
  Play,
  Plus,
  Settings,
  Star,
  StopCircle,
  Target,
  TrendingUp,
  User,
  Zap,
} from "lucide-react";

type Priority = "Deep" | "Normal" | "Quick";

type TaskItem = {
  id: number;
  title: string;
  priority: Priority;
  duration: number;
  completed: boolean;
  tag: string;
  estimatedTime: number;
  dueDate: string;
  inProgress?: boolean;
};

type Profile = {
  name: string;
  email: string;
  studyGoals: string;
  focusTime: string;
  tasksCompleted: number;
  totalStudyHours: number;
  streak: number;
};

type LoginForm = { email: string; password: string };
type SignupForm = {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  studyGoals: string;
  focusTime: string;
};

export default function LightModePage() {
  const [currentView, setCurrentView] = useState<"main" | "profile" | "login" | "signup">("main");
  const [isLoggedIn, setIsLoggedIn] = useState(true);
  const [userProfile, setUserProfile] = useState<Profile>({
    name: "Agnes Chen",
    email: "agnes@example.com",
    studyGoals: "Prepare for final exams",
    focusTime: "morning",
    tasksCompleted: 47,
    totalStudyHours: 128,
    streak: 12,
  });

  const [showPassword, setShowPassword] = useState(false);
  const [loginForm, setLoginForm] = useState<LoginForm>({ email: "", password: "" });
  const [signupForm, setSignupForm] = useState<SignupForm>({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    studyGoals: "",
    focusTime: "morning",
  });

  const [tasks, setTasks] = useState<TaskItem[]>([
    { id: 1, title: "Review math chapter 3", priority: "Deep", duration: 50, completed: true, tag: "Top3", estimatedTime: 50, dueDate: "2026-01-13" },
    { id: 2, title: "Summarize history notes", priority: "Normal", duration: 30, completed: false, tag: "Top3", estimatedTime: 30, dueDate: "2026-01-14", inProgress: true },
    { id: 3, title: "English vocab drill", priority: "Normal", duration: 20, completed: false, tag: "Top3", estimatedTime: 20, dueDate: "2026-01-15" },
    { id: 4, title: "Flashcards sprint", priority: "Quick", duration: 8, completed: false, tag: "", estimatedTime: 8, dueDate: "2026-01-16" },
    { id: 5, title: "Physics homework", priority: "Deep", duration: 60, completed: true, tag: "", estimatedTime: 60, dueDate: "2026-01-12" },
    { id: 6, title: "Biology lab report", priority: "Normal", duration: 45, completed: false, tag: "", estimatedTime: 45, dueDate: "2026-01-17" },
    { id: 7, title: "Chemistry practice", priority: "Quick", duration: 15, completed: true, tag: "", estimatedTime: 15, dueDate: "2026-01-11" },
  ]);

  const [newTask, setNewTask] = useState("");
  const [selectedPriority, setSelectedPriority] = useState<Priority>("Normal");
  const [newTaskDueDate, setNewTaskDueDate] = useState("");
  const [newTaskDuration, setNewTaskDuration] = useState<number>(30);
  const [activeTask, setActiveTask] = useState<TaskItem | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackTask, setFeedbackTask] = useState<TaskItem | null>(null);
  const [rating, setRating] = useState(0);
  const [eegConnected, setEegConnected] = useState(false);
  const [viewMode, setViewMode] = useState<"list" | "cognitive">("cognitive");
  const [swipedTask, setSwipedTask] = useState<number | null>(null);
  const [draggedTask, setDraggedTask] = useState<TaskItem | null>(null);
  const [dragOverTask, setDragOverTask] = useState<TaskItem | null>(null);
  const [showSchedule, setShowSchedule] = useState(false);

  const toggleTask = (id: number) => {
    setTasks((prev) => prev.map((task) => (task.id === id ? { ...task, completed: !task.completed } : task)));
  };

  const moveToTop3 = (id: number) => {
    setTasks((prev) => prev.map((task) => (task.id === id ? { ...task, tag: "Top3" } : task)));
    setSwipedTask(null);
  };

  const removeFromTop3 = (id: number) => {
    setTasks((prev) => prev.map((task) => (task.id === id ? { ...task, tag: "" } : task)));
  };

  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, task: TaskItem) => {
    setDraggedTask(task);
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>, task: TaskItem) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    if (draggedTask && task.id !== draggedTask.id) {
      setDragOverTask(task);
    }
  };

  const handleDragLeave = () => {
    setDragOverTask(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>, targetTask: TaskItem) => {
    e.preventDefault();
    if (!draggedTask || draggedTask.id === targetTask.id) return;

    const newTasks = [...tasks];
    const draggedIndex = newTasks.findIndex((t) => t.id === draggedTask.id);
    const targetIndex = newTasks.findIndex((t) => t.id === targetTask.id);

    const [removed] = newTasks.splice(draggedIndex, 1);
    newTasks.splice(targetIndex, 0, removed);

    setTasks(newTasks);
    setDraggedTask(null);
    setDragOverTask(null);
  };

  const handleDragEnd = () => {
    setDraggedTask(null);
    setDragOverTask(null);
  };

  const addTask = () => {
    if (newTask.trim() && newTaskDueDate) {
      setTasks((prev) => [
        ...prev,
        {
          id: Date.now(),
          title: newTask,
          priority: selectedPriority,
          duration: newTaskDuration,
          completed: false,
          tag: "",
          estimatedTime: newTaskDuration,
          dueDate: newTaskDueDate,
        },
      ]);
      setNewTask("");
      setNewTaskDueDate("");
      setNewTaskDuration(30);
    }
  };

  const startTask = (task: TaskItem) => {
    setActiveTask(task);
    setElapsedTime(0);
    setIsPaused(false);
    setTasks((prev) => prev.map((t) => (t.id === task.id ? { ...t, inProgress: true } : t)));
  };

  const pauseTask = () => {
    setIsPaused((prev) => !prev);
  };

  const stopTask = () => {
    if (!activeTask) return;
    setFeedbackTask(activeTask);
    setShowFeedback(true);
    setTasks((prev) => prev.map((t) => (t.id === activeTask.id ? { ...t, inProgress: false } : t)));
    setActiveTask(null);
    setElapsedTime(0);
  };

  const submitFeedback = () => {
    if (!feedbackTask) return;
    const feedbackData = {
      taskId: feedbackTask.id,
      taskTitle: feedbackTask.title,
      priority: feedbackTask.priority,
      estimatedTime: feedbackTask.estimatedTime,
      actualTime: elapsedTime,
      efficiency: rating,
      timestamp: new Date().toISOString(),
      timeOfDay: new Date().getHours(),
      eegData: eegConnected ? "EEG data captured" : null,
    };

    console.log("Submitting feedback to AI:", feedbackData);

    setShowFeedback(false);
    setRating(0);
    setFeedbackTask(null);
    toggleTask(feedbackData.taskId);
  };

  const getPriorityColor = (priority: Priority) => {
    switch (priority) {
      case "Deep":
        return "bg-purple-100 text-purple-700 border-purple-200";
      case "Normal":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "Quick":
        return "bg-green-100 text-green-700 border-green-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;
    if (activeTask && !isPaused) {
      interval = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeTask, isPaused]);

  const topTasks = useMemo(() => tasks.filter((t) => t.tag === "Top3" && !t.completed).slice(0, 3), [tasks]);
  const completedCount = useMemo(() => tasks.filter((t) => t.completed).length, [tasks]);

  const getTasksByLoad = () => {
    const activeTasks = tasks.filter((t) => !t.completed);
    return {
      deep: activeTasks.filter((t) => t.priority === "Deep"),
      normal: activeTasks.filter((t) => t.priority === "Normal"),
      quick: activeTasks.filter((t) => t.priority === "Quick"),
    };
  };

  const tasksByLoad = getTasksByLoad();

  const getWeekDates = () => {
    const today = new Date(2026, 0, 17);
    const sunday = new Date(today);
    sunday.setDate(today.getDate() - today.getDay());

    const weekDates: Date[] = [];
    for (let i = 0; i < 7; i++) {
      const date = new Date(sunday);
      date.setDate(sunday.getDate() + i);
      weekDates.push(date);
    }
    return weekDates;
  };

  const weekDates = getWeekDates();

  const getTasksForDate = (date: Date) => {
    const dateStr = date.toISOString().split("T")[0];
    return tasks.filter((task) => task.dueDate === dateStr);
  };

  const getDateStats = (date: Date) => {
    const dateTasks = getTasksForDate(date);
    const completed = dateTasks.filter((t) => t.completed).length;
    const inProgress = dateTasks.filter((t) => t.inProgress && !t.completed).length;
    const pending = dateTasks.filter((t) => !t.completed && !t.inProgress).length;
    return { completed, inProgress, pending, total: dateTasks.length };
  };

  const generateAISchedule = () => {
    const activeTasks = tasks.filter((t) => !t.completed);
    const deepTasks = activeTasks.filter((t) => t.priority === "Deep");
    const normalTasks = activeTasks.filter((t) => t.priority === "Normal");
    const quickTasks = activeTasks.filter((t) => t.priority === "Quick");

    const schedule: {
      time: string;
      task: string;
      priority: Priority;
      duration: number;
      type: "deep" | "normal";
      taskId: number;
    }[] = [];

    if (deepTasks.length > 0) {
      let remainingTime = 120;
      let currentTime = "9:00";

      for (const task of deepTasks.slice(0, 3)) {
        if (remainingTime >= task.duration) {
          const endHour = 9 + Math.floor((120 - remainingTime + task.duration) / 60);
          const endMin = (120 - remainingTime + task.duration) % 60;
          const endTime = `${endHour}:${endMin.toString().padStart(2, "0")}`;

          schedule.push({
            time: `${currentTime} - ${endTime}`,
            task: task.title,
            priority: task.priority,
            duration: task.duration,
            type: "deep",
            taskId: task.id,
          });

          remainingTime -= task.duration;
          currentTime = endTime;
        }
      }
    }

    const afternoonTasks = [...normalTasks.slice(0, 2), ...quickTasks.slice(0, 2)];
    if (afternoonTasks.length > 0) {
      let currentMinute = 0;

      for (const task of afternoonTasks) {
        const startHour = 14 + Math.floor(currentMinute / 60);
        const startMin = currentMinute % 60;
        const endMinute = currentMinute + task.duration;
        const endHour = 14 + Math.floor(endMinute / 60);
        const endMin = endMinute % 60;

        schedule.push({
          time: `${startHour}:${startMin.toString().padStart(2, "0")} - ${endHour}:${endMin.toString().padStart(2, "0")}`,
          task: task.title,
          priority: task.priority,
          duration: task.duration,
          type: "normal",
          taskId: task.id,
        });

        currentMinute = endMinute;
        if (currentMinute >= 120) break;
      }
    }

    return schedule;
  };

  const aiSchedule = generateAISchedule();

  const applyScheduleToCalendar = () => {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const tomorrowStr = tomorrow.toISOString().split("T")[0];

    const updatedTasks = tasks.map((task) => {
      const scheduleItem = aiSchedule.find((s) => s.taskId === task.id);
      if (scheduleItem) {
        return { ...task, dueDate: tomorrowStr, tag: "Top3" };
      }
      return task;
    });

    setTasks(updatedTasks);
    setShowSchedule(false);
    alert("✅ Schedule applied to tomorrow!");
  };

  const handleLogin = () => {
    setIsLoggedIn(true);
    setUserProfile((prev) => ({ ...prev, email: loginForm.email }));
    setCurrentView("main");
  };

  const handleSignup = () => {
    if (signupForm.password !== signupForm.confirmPassword) {
      alert("Passwords do not match!");
      return;
    }
    setUserProfile({
      name: signupForm.name,
      email: signupForm.email,
      studyGoals: signupForm.studyGoals,
      focusTime: signupForm.focusTime,
      tasksCompleted: 0,
      totalStudyHours: 0,
      streak: 0,
    });
    setIsLoggedIn(true);
    setCurrentView("main");
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setCurrentView("login");
  };

  if (!isLoggedIn) {
    if (currentView === "signup") {
      return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-purple-50 flex items-center justify-center p-4">
          <div className="max-w-md w-full">
            <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <Zap className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Create Account</h2>
                <p className="text-gray-600">Start optimizing your study time with AI</p>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="text"
                      value={signupForm.name}
                      onChange={(e) => setSignupForm({ ...signupForm, name: e.target.value })}
                      className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="John Doe"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="email"
                      value={signupForm.email}
                      onChange={(e) => setSignupForm({ ...signupForm, email: e.target.value })}
                      className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="your.email@example.com"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type={showPassword ? "text" : "password"}
                      value={signupForm.password}
                      onChange={(e) => setSignupForm({ ...signupForm, password: e.target.value })}
                      className="w-full pl-11 pr-12 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="••••••••"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((prev) => !prev)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Confirm Password</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type={showPassword ? "text" : "password"}
                      value={signupForm.confirmPassword}
                      onChange={(e) => setSignupForm({ ...signupForm, confirmPassword: e.target.value })}
                      className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="••••••••"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Study Goals</label>
                  <textarea
                    value={signupForm.studyGoals}
                    onChange={(e) => setSignupForm({ ...signupForm, studyGoals: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., Prepare for SAT, Learn Python..."
                    rows={2}
                  />
                </div>

                <button
                  onClick={handleSignup}
                  className="w-full py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white font-bold rounded-xl hover:from-purple-600 hover:to-purple-700 transition-all shadow-md"
                >
                  Create Account
                </button>
              </div>

              <div className="mt-6 text-center">
                <p className="text-gray-600">
                  Already have an account?{" "}
                  <button
                    onClick={() => setCurrentView("login")}
                    className="text-purple-600 hover:text-purple-700 font-semibold"
                  >
                    Login
                  </button>
                </p>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-purple-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full">
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Zap className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-2">Welcome Back</h2>
              <p className="text-gray-600">Login to continue your learning journey</p>
            </div>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="email"
                    value={loginForm.email}
                    onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
                    className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="your.email@example.com"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type={showPassword ? "text" : "password"}
                    value={loginForm.password}
                    onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                    className="w-full pl-11 pr-12 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((prev) => !prev)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <button
                onClick={handleLogin}
                className="w-full py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white font-bold rounded-xl hover:from-purple-600 hover:to-purple-700 transition-all shadow-md"
              >
                Login
              </button>
            </div>

            <div className="mt-6 text-center">
              <p className="text-gray-600">
                Don't have an account?{" "}
                <button
                  onClick={() => setCurrentView("signup")}
                  className="text-purple-600 hover:text-purple-700 font-semibold"
                >
                  Sign up
                </button>
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (currentView === "profile") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-purple-50">
        <div className="bg-white border-b border-gray-200 px-8 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">NeuroPlan</h1>
                <p className="text-sm text-gray-500">User Profile</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setCurrentView("main")}
                className="px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-all font-medium"
              >
                Back to Dashboard
              </button>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-all"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm font-medium">Logout</span>
              </button>
            </div>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-8 py-8">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="md:col-span-1">
              <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
                <div className="text-center mb-6">
                  <div className="w-24 h-24 bg-gradient-to-br from-purple-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-3xl font-bold text-white">
                      {userProfile.name
                        .split(" ")
                        .map((n) => n[0])
                        .join("")}
                    </span>
                  </div>
                  <h2 className="text-xl font-bold text-gray-900 mb-1">{userProfile.name}</h2>
                  <p className="text-gray-600 text-sm">{userProfile.email}</p>
                </div>

                <div className="space-y-4">
                  <div className="p-4 bg-purple-50 rounded-xl border border-purple-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-purple-900">Study Streak</span>
                      <span className="text-2xl">🔥</span>
                    </div>
                    <div className="text-3xl font-bold text-purple-600">{userProfile.streak} days</div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-blue-50 rounded-xl border border-blue-200">
                      <div className="text-xs text-blue-700 mb-1">Tasks Done</div>
                      <div className="text-xl font-bold text-blue-600">{userProfile.tasksCompleted}</div>
                    </div>
                    <div className="p-3 bg-green-50 rounded-xl border border-green-200">
                      <div className="text-xs text-green-700 mb-1">Study Hours</div>
                      <div className="text-xl font-bold text-green-600">{userProfile.totalStudyHours}h</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="md:col-span-2 space-y-6">
              <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-2 mb-4">
                  <Settings className="w-5 h-5 text-purple-600" />
                  <h3 className="font-semibold text-gray-900">Study Goals</h3>
                </div>
                <textarea
                  value={userProfile.studyGoals}
                  onChange={(e) => setUserProfile({ ...userProfile, studyGoals: e.target.value })}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                  rows={3}
                />
              </div>

              <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
                <h3 className="font-semibold text-gray-900 mb-4">Preferences</h3>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Peak Focus Time</label>
                  <select
                    value={userProfile.focusTime}
                    onChange={(e) => setUserProfile({ ...userProfile, focusTime: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="morning">Morning (6 AM - 12 PM)</option>
                    <option value="afternoon">Afternoon (12 PM - 6 PM)</option>
                    <option value="evening">Evening (6 PM - 12 AM)</option>
                    <option value="night">Night (12 AM - 6 AM)</option>
                  </select>
                </div>
              </div>

              <button className="w-full py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white font-semibold rounded-xl hover:from-purple-600 hover:to-purple-700 transition-all shadow-md">
                Save Changes
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-purple-50">
      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">NeuroPlan</h1>
              <p className="text-sm text-gray-500">AI Planning System</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setCurrentView("profile")}
              className="flex items-center gap-2 px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-xl border border-purple-200 transition-all"
            >
              <User className="w-4 h-4" />
              <span className="text-sm font-medium">{userProfile.name}</span>
            </button>
            <button
              onClick={() => setEegConnected((prev) => !prev)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all ${
                eegConnected
                  ? "bg-green-50 border-green-300 text-green-700"
                  : "bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100"
              }`}
            >
              <Brain className="w-4 h-4" />
              <span className="text-sm font-medium">
                {eegConnected ? "EEG Connected" : "Connect EEG"}
              </span>
              {eegConnected && <Activity className="w-4 h-4 animate-pulse" />}
            </button>
            <div className="text-right">
              <div className="text-sm text-gray-500">Today's Progress</div>
              <div className="text-xl font-bold text-purple-600">
                {completedCount}/{tasks.length}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white rounded-2xl p-5 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                    <CheckCircle2 className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900">{completedCount}</div>
                    <div className="text-sm text-gray-500">Completed</div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl p-5 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                    <Target className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900">{topTasks.length}</div>
                    <div className="text-sm text-gray-500">Top Priority</div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl p-5 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-green-600" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900">85%</div>
                    <div className="text-sm text-gray-500">Avg Efficiency</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl p-6 text-white shadow-lg">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="w-6 h-6" />
                <h2 className="text-xl font-bold">Today Top 3</h2>
              </div>
              <p className="text-purple-100 text-sm mb-6">Focus on these high-impact tasks</p>
              <div className="space-y-3">
                {topTasks.map((task, idx) => (
                  <div
                    key={task.id}
                    draggable
                    onDragStart={(e) => handleDragStart(e, task)}
                    onDragOver={(e) => handleDragOver(e, task)}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, task)}
                    onDragEnd={handleDragEnd}
                    className={`bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20 transition-all cursor-move ${
                      dragOverTask?.id === task.id ? "border-yellow-300 border-2 scale-105" : ""
                    } ${draggedTask?.id === task.id ? "opacity-50" : ""}`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center font-bold flex-shrink-0">
                        {idx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium mb-1 flex items-center gap-2">
                          {task.title}
                          <span className="text-xs opacity-70">🔀 Drag to reorder</span>
                        </div>
                        <div className="text-sm text-purple-100 flex items-center gap-2 mb-3">
                          <Clock className="w-4 h-4" />
                          {task.duration}m · {task.priority}
                        </div>
                        {activeTask?.id === task.id ? (
                          <div className="bg-white/20 rounded-lg p-3 border border-white/30">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm font-medium">Recording...</span>
                              <span className="text-lg font-bold font-mono">{formatTime(elapsedTime)}</span>
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={pauseTask}
                                className="flex-1 bg-white/20 hover:bg-white/30 py-2 rounded-lg flex items-center justify-center gap-2 transition-all"
                              >
                                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                                <span className="text-sm font-medium">{isPaused ? "Resume" : "Pause"}</span>
                              </button>
                              <button
                                onClick={stopTask}
                                className="flex-1 bg-red-500/80 hover:bg-red-500 py-2 rounded-lg flex items-center justify-center gap-2 transition-all"
                              >
                                <StopCircle className="w-4 h-4" />
                                <span className="text-sm font-medium">Finish</span>
                              </button>
                            </div>
                          </div>
                        ) : (
                          <button
                            onClick={() => startTask(task)}
                            disabled={activeTask !== null}
                            className="w-full bg-white hover:bg-purple-50 text-purple-600 py-2.5 rounded-lg flex items-center justify-center gap-2 font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <Play className="w-4 h-4" />
                            Start Task
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {topTasks.length === 0 && (
                  <div className="text-center py-8 text-purple-100">
                    <Target className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No top priority tasks yet</p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Add New Task</h2>
              <div className="space-y-4">
                <input
                  type="text"
                  placeholder="e.g., Finish math workbook"
                  value={newTask}
                  onChange={(e) => setNewTask(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") addTask();
                  }}
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
                <div className="grid grid-cols-2 gap-3">
                  <select
                    value={selectedPriority}
                    onChange={(e) => setSelectedPriority(e.target.value as Priority)}
                    className="px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="Deep">Deep Focus</option>
                    <option value="Normal">Normal</option>
                    <option value="Quick">Quick Task</option>
                  </select>
                  <input
                    type="number"
                    placeholder="Duration (min)"
                    value={newTaskDuration}
                    onChange={(e) => setNewTaskDuration(Number(e.target.value) || 0)}
                    min={1}
                    className="px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div className="flex gap-3">
                  <input
                    type="date"
                    value={newTaskDueDate}
                    onChange={(e) => setNewTaskDueDate(e.target.value)}
                    min={new Date().toISOString().split("T")[0]}
                    className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <button
                    onClick={addTask}
                    disabled={!newTask.trim() || !newTaskDueDate}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl hover:from-purple-600 hover:to-purple-700 transition-all flex items-center gap-2 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Plus className="w-5 h-5" />
                    Add Task
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-purple-600" />
                  All Tasks ({tasks.filter((t) => !t.completed).length})
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setViewMode("list")}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      viewMode === "list" ? "bg-purple-100 text-purple-700" : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}
                  >
                    List View
                  </button>
                  <button
                    onClick={() => setViewMode("cognitive")}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      viewMode === "cognitive" ? "bg-purple-100 text-purple-700" : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}
                  >
                    By Load
                  </button>
                </div>
              </div>

              {viewMode === "list" ? (
                <div className="space-y-3">
                  {tasks.map((task) => (
                    <div
                      key={task.id}
                      className={`p-4 rounded-xl border transition-all ${
                        task.completed
                          ? "bg-blue-50 border-blue-200"
                          : task.inProgress
                          ? "bg-yellow-50 border-yellow-200"
                          : "bg-white border-gray-200 hover:border-purple-300 hover:shadow-sm"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <button onClick={() => toggleTask(task.id)} className="mt-0.5">
                          {task.completed ? (
                            <CheckCircle2 className="w-5 h-5 text-blue-600" />
                          ) : (
                            <Circle className="w-5 h-5 text-gray-400 hover:text-purple-500" />
                          )}
                        </button>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <span
                              className={`text-sm font-medium ${
                                task.completed
                                  ? "line-through text-blue-700"
                                  : task.inProgress
                                  ? "text-yellow-900"
                                  : "text-gray-900"
                              }`}
                            >
                              {task.title}
                            </span>
                            {task.tag && (
                              <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
                                {task.tag}
                              </span>
                            )}
                            {task.inProgress && !task.completed && (
                              <span className="px-2 py-0.5 bg-yellow-200 text-yellow-800 text-xs rounded-full font-medium">
                                In Progress
                              </span>
                            )}
                            {task.completed && (
                              <span className="px-2 py-0.5 bg-blue-200 text-blue-800 text-xs rounded-full font-medium">
                                Completed
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-500">
                            <span className={`px-3 py-1 rounded-lg border text-xs font-medium ${getPriorityColor(task.priority)}`}>
                              {task.priority}
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="w-4 h-4" />
                              {task.duration}m
                            </span>
                            {task.dueDate && (
                              <span className="flex items-center gap-1 text-orange-600">
                                <Calendar className="w-4 h-4" />
                                {new Date(task.dueDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="border-2 border-purple-300 rounded-xl p-4 bg-gradient-to-br from-purple-50 to-white shadow-md">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
                        <Brain className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-bold text-purple-900">Deep Focus Zone</h3>
                        <p className="text-xs text-purple-700">Best for peak mental hours • {tasksByLoad.deep.length} tasks</p>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-purple-900">
                          {tasksByLoad.deep.reduce((sum, t) => sum + t.duration, 0)}m
                        </div>
                        <div className="text-xs text-purple-600">total time</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      {tasksByLoad.deep.length > 0 ? (
                        tasksByLoad.deep.map((task) => (
                          <div key={task.id} className="relative overflow-hidden">
                            {swipedTask === task.id && (
                              <div className="absolute inset-0 bg-gradient-to-l from-yellow-400 to-yellow-300 flex items-center justify-end px-4 rounded-lg">
                                <div className="flex items-center gap-2 text-yellow-900 font-medium">
                                  <Star className="w-5 h-5 fill-yellow-900" />
                                  <span>Add to Top 3</span>
                                </div>
                              </div>
                            )}

                            <div
                              className={`p-3 rounded-lg border-2 transition-all relative ${
                                task.inProgress
                                  ? "bg-yellow-50 border-yellow-300"
                                  : "bg-white border-purple-200 hover:border-purple-400 hover:shadow-sm"
                              } ${swipedTask === task.id ? "translate-x-[-120px]" : ""}`}
                              style={{ transition: "transform 0.3s ease" }}
                            >
                              <div className="flex items-start gap-3">
                                <button onClick={() => toggleTask(task.id)} className="mt-0.5">
                                  <Circle className="w-4 h-4 text-purple-500 hover:text-purple-700" />
                                </button>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-sm font-semibold text-purple-900">{task.title}</span>
                                    {task.tag === "Top3" && (
                                      <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
                                        Top3
                                      </span>
                                    )}
                                    {task.inProgress && (
                                      <span className="px-2 py-0.5 bg-yellow-200 text-yellow-800 text-xs rounded-full font-medium">
                                        In Progress
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-3 text-xs text-purple-700">
                                    <span className="flex items-center gap-1 font-medium">
                                      <Clock className="w-3 h-3" />
                                      {task.duration}m
                                    </span>
                                    {task.dueDate && (
                                      <span className="flex items-center gap-1">
                                        <Calendar className="w-3 h-3" />
                                        {new Date(task.dueDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                {task.tag !== "Top3" ? (
                                  <button
                                    onClick={() => (swipedTask === task.id ? moveToTop3(task.id) : setSwipedTask(task.id))}
                                    className="flex items-center gap-1 px-3 py-1.5 bg-yellow-100 hover:bg-yellow-200 text-yellow-700 rounded-lg text-xs font-medium transition-all"
                                  >
                                    <Star className="w-3 h-3" />
                                    Top 3
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => removeFromTop3(task.id)}
                                    className="flex items-center gap-1 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-xs font-medium transition-all"
                                  >
                                    Remove
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-4 text-purple-600 text-sm">No deep focus tasks • Great job! 🎉</div>
                      )}
                    </div>
                  </div>

                  <div className="border border-blue-200 rounded-xl p-4 bg-gradient-to-br from-blue-50 to-white">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                        <Target className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-bold text-blue-900">Normal Tasks</h3>
                        <p className="text-xs text-blue-700">Moderate focus required • {tasksByLoad.normal.length} tasks</p>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-blue-900">
                          {tasksByLoad.normal.reduce((sum, t) => sum + t.duration, 0)}m
                        </div>
                        <div className="text-xs text-blue-600">total time</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      {tasksByLoad.normal.length > 0 ? (
                        tasksByLoad.normal.map((task) => (
                          <div key={task.id} className="relative overflow-hidden">
                            {swipedTask === task.id && (
                              <div className="absolute inset-0 bg-gradient-to-l from-yellow-400 to-yellow-300 flex items-center justify-end px-4 rounded-lg">
                                <div className="flex items-center gap-2 text-yellow-900 font-medium">
                                  <Star className="w-5 h-5 fill-yellow-900" />
                                  <span>Add to Top 3</span>
                                </div>
                              </div>
                            )}

                            <div
                              className={`p-3 rounded-lg border transition-all relative ${
                                task.inProgress
                                  ? "bg-yellow-50 border-yellow-300"
                                  : "bg-white border-blue-200 hover:border-blue-300 hover:shadow-sm"
                              } ${swipedTask === task.id ? "translate-x-[-120px]" : ""}`}
                              style={{ transition: "transform 0.3s ease" }}
                            >
                              <div className="flex items-start gap-3">
                                <button onClick={() => toggleTask(task.id)} className="mt-0.5">
                                  <Circle className="w-4 h-4 text-blue-500 hover:text-blue-700" />
                                </button>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-sm font-medium text-blue-900">{task.title}</span>
                                    {task.tag === "Top3" && (
                                      <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
                                        Top3
                                      </span>
                                    )}
                                    {task.inProgress && (
                                      <span className="px-2 py-0.5 bg-yellow-200 text-yellow-800 text-xs rounded-full font-medium">
                                        In Progress
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-3 text-xs text-blue-700">
                                    <span className="flex items-center gap-1">
                                      <Clock className="w-3 h-3" />
                                      {task.duration}m
                                    </span>
                                    {task.dueDate && (
                                      <span className="flex items-center gap-1">
                                        <Calendar className="w-3 h-3" />
                                        {new Date(task.dueDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                {task.tag !== "Top3" ? (
                                  <button
                                    onClick={() => (swipedTask === task.id ? moveToTop3(task.id) : setSwipedTask(task.id))}
                                    className="flex items-center gap-1 px-3 py-1.5 bg-yellow-100 hover:bg-yellow-200 text-yellow-700 rounded-lg text-xs font-medium transition-all"
                                  >
                                    <Star className="w-3 h-3" />
                                    Top 3
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => removeFromTop3(task.id)}
                                    className="flex items-center gap-1 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-xs font-medium transition-all"
                                  >
                                    Remove
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-4 text-blue-600 text-sm">No normal tasks</div>
                      )}
                    </div>
                  </div>

                  <div className="border border-green-200 rounded-xl p-4 bg-gradient-to-br from-green-50 to-white">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
                        <Zap className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-bold text-green-900">Quick Wins</h3>
                        <p className="text-xs text-green-700">Fill the gaps • Perfect for breaks • {tasksByLoad.quick.length} tasks</p>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-green-900">
                          {tasksByLoad.quick.reduce((sum, t) => sum + t.duration, 0)}m
                        </div>
                        <div className="text-xs text-green-600">total time</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      {tasksByLoad.quick.length > 0 ? (
                        tasksByLoad.quick.map((task) => (
                          <div key={task.id} className="relative overflow-hidden">
                            {swipedTask === task.id && (
                              <div className="absolute inset-0 bg-gradient-to-l from-yellow-400 to-yellow-300 flex items-center justify-end px-4 rounded-lg">
                                <div className="flex items-center gap-2 text-yellow-900 font-medium">
                                  <Star className="w-5 h-5 fill-yellow-900" />
                                  <span>Add to Top 3</span>
                                </div>
                              </div>
                            )}

                            <div
                              className={`p-3 rounded-lg border transition-all relative ${
                                task.inProgress
                                  ? "bg-yellow-50 border-yellow-300"
                                  : "bg-white border-green-200 hover:border-green-300 hover:shadow-sm"
                              } ${swipedTask === task.id ? "translate-x-[-120px]" : ""}`}
                              style={{ transition: "transform 0.3s ease" }}
                            >
                              <div className="flex items-start gap-3">
                                <button onClick={() => toggleTask(task.id)} className="mt-0.5">
                                  <Circle className="w-4 h-4 text-green-500 hover:text-green-700" />
                                </button>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-sm font-medium text-green-900">{task.title}</span>
                                    {task.tag === "Top3" && (
                                      <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
                                        Top3
                                      </span>
                                    )}
                                    {task.inProgress && (
                                      <span className="px-2 py-0.5 bg-yellow-200 text-yellow-800 text-xs rounded-full font-medium">
                                        In Progress
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-3 text-xs text-green-700">
                                    <span className="flex items-center gap-1">
                                      <Clock className="w-3 h-3" />
                                      {task.duration}m
                                    </span>
                                    {task.dueDate && (
                                      <span className="flex items-center gap-1">
                                        <Calendar className="w-3 h-3" />
                                        {new Date(task.dueDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                {task.tag !== "Top3" ? (
                                  <button
                                    onClick={() => (swipedTask === task.id ? moveToTop3(task.id) : setSwipedTask(task.id))}
                                    className="flex items-center gap-1 px-3 py-1.5 bg-yellow-100 hover:bg-yellow-200 text-yellow-700 rounded-lg text-xs font-medium transition-all"
                                  >
                                    <Star className="w-3 h-3" />
                                    Top 3
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => removeFromTop3(task.id)}
                                    className="flex items-center gap-1 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-xs font-medium transition-all"
                                  >
                                    Remove
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-4 text-green-600 text-sm">No quick tasks</div>
                      )}
                    </div>
                  </div>

                  {completedCount > 0 && (
                    <div className="border border-blue-200 rounded-xl p-4 bg-blue-50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="w-5 h-5 text-blue-600" />
                          <span className="font-medium text-blue-900">
                            {completedCount} task{completedCount > 1 ? "s" : ""} completed today
                          </span>
                        </div>
                        <span className="text-xl">🎉</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-600" />
                  <h3 className="font-semibold text-gray-900">AI Insights</h3>
                </div>
                <button
                  onClick={() => setShowSchedule((prev) => !prev)}
                  className="text-xs px-3 py-1.5 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg font-medium transition-all"
                >
                  {showSchedule ? "Hide" : "View Schedule"}
                </button>
              </div>

              {!showSchedule ? (
                <div className="space-y-3">
                  <div className="p-3 bg-purple-50 rounded-lg border border-purple-100">
                    <div className="text-sm font-medium text-purple-900 mb-1">Optimal Deep Work</div>
                    <div className="text-xs text-purple-700">9:00 AM - 11:00 AM</div>
                    <div className="text-xs text-purple-600 mt-2">Based on your performance data</div>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
                    <div className="text-sm font-medium text-blue-900 mb-1">Current Focus Level</div>
                    <div className="text-xs text-blue-700">High {eegConnected && "(EEG: Alpha waves detected)"}</div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="text-center mb-4">
                    <h4 className="font-bold text-gray-900 text-lg mb-1">Tomorrow's Suggested Schedule</h4>
                    <p className="text-xs text-gray-600">AI-optimized based on your cognitive patterns</p>
                  </div>

                  <div className="relative pl-8 space-y-4">
                    <div className="absolute left-3 top-2 bottom-2 w-0.5 bg-gradient-to-b from-purple-500 via-blue-400 to-green-400" />

                    {aiSchedule.length > 0 ? (
                      aiSchedule.map((item, idx) => (
                        <div key={idx} className="relative">
                          <div
                            className={`absolute -left-[26px] w-5 h-5 rounded-full border-2 ${
                              item.type === "deep" ? "bg-purple-500 border-purple-300" : "bg-blue-400 border-blue-300"
                            }`}
                          />

                          <div
                            className={`p-4 rounded-xl border-2 transition-all ${
                              item.type === "deep"
                                ? "bg-gradient-to-br from-purple-50 to-purple-100 border-purple-300 shadow-md"
                                : "bg-gradient-to-br from-blue-50 to-white border-blue-200"
                            }`}
                          >
                            <div className="flex items-start justify-between mb-2">
                              <div className={`font-bold text-sm ${item.type === "deep" ? "text-purple-900" : "text-blue-900"}`}>
                                {item.time}
                              </div>
                              {item.type === "deep" && (
                                <span className="px-2 py-0.5 bg-yellow-200 text-yellow-800 text-xs rounded-full font-bold">
                                  ⚡ Peak Focus
                                </span>
                              )}
                            </div>
                            <div className={`font-semibold mb-1 ${item.type === "deep" ? "text-purple-900" : "text-blue-800"}`}>
                              {item.task}
                            </div>
                            <div className="flex items-center gap-3 text-xs text-gray-600">
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {item.duration}m
                              </span>
                              <span
                                className={`px-2 py-0.5 rounded-full font-medium ${
                                  item.priority === "Deep"
                                    ? "bg-purple-100 text-purple-700"
                                    : item.priority === "Normal"
                                    ? "bg-blue-100 text-blue-700"
                                    : "bg-green-100 text-green-700"
                                }`}
                              >
                                {item.priority}
                              </span>
                            </div>
                            {item.type === "deep" && (
                              <div className="mt-2 text-xs text-purple-700 flex items-center gap-1">
                                <Brain className="w-3 h-3" />
                                <span>Matches your high-performance window</span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-6 text-gray-500">
                        <Target className="w-10 h-10 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No tasks to schedule</p>
                      </div>
                    )}
                  </div>

                  {aiSchedule.length > 0 && (
                    <div className="pt-4 border-t border-gray-200">
                      <div className="flex items-start gap-2 mb-4 p-3 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg">
                        <Activity className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
                        <div className="text-xs text-gray-700">
                          <span className="font-semibold">AI Analysis:</span> This schedule maximizes your productivity by placing
                          <span className="font-semibold text-purple-700"> {aiSchedule.filter((s) => s.type === "deep").length} deep focus tasks</span> during
                          your peak mental performance hours (9-11 AM)
                          {eegConnected && ", validated by your EEG patterns"}.
                        </div>
                      </div>

                      <button
                        onClick={applyScheduleToCalendar}
                        className="w-full py-3 bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white font-semibold rounded-xl transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2"
                      >
                        <CheckCircle2 className="w-5 h-5" />
                        Apply to Tomorrow's Calendar
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">This Week</h3>
                <div className="flex gap-2">
                  <button className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors">
                    <ChevronLeft className="w-4 h-4 text-gray-600" />
                  </button>
                  <button className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors">
                    <ChevronRight className="w-4 h-4 text-gray-600" />
                  </button>
                </div>
              </div>
              <div className="text-center text-sm text-gray-600 mb-4">Jan 11 – 17, 2026</div>

              <div className="flex items-center justify-center gap-3 mb-4 text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <span className="text-gray-600">Completed</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded-full bg-yellow-500" />
                  <span className="text-gray-600">In Progress</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded-full bg-gray-400" />
                  <span className="text-gray-600">Pending</span>
                </div>
              </div>

              <div className="space-y-2">
                {weekDates.map((date, idx) => {
                  const dayTasks = getTasksForDate(date);
                  const stats = getDateStats(date);
                  const isToday = idx === 6;
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border transition-all ${
                        isToday ? "bg-purple-50 border-purple-200" : "bg-gray-50 border-gray-200 hover:bg-gray-100"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <div className={`text-xs font-medium ${isToday ? "text-purple-600" : "text-gray-500"}`}>
                            {date.toLocaleDateString("en-US", { weekday: "short" })}
                          </div>
                          <div className={`text-sm font-bold ${isToday ? "text-purple-700" : "text-gray-900"}`}>
                            {date.getDate()}
                          </div>
                        </div>
                        {stats.total > 0 && (
                          <div className="flex items-center gap-1">
                            {stats.completed > 0 && (
                              <div className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
                                {stats.completed}
                              </div>
                            )}
                            {stats.inProgress > 0 && (
                              <div className="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700">
                                {stats.inProgress}
                              </div>
                            )}
                            {stats.pending > 0 && (
                              <div className="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-200 text-gray-700">
                                {stats.pending}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      {dayTasks.length > 0 && (
                        <div className="space-y-1.5">
                          {dayTasks.slice(0, 3).map((task) => (
                            <div
                              key={task.id}
                              className={`text-xs p-2 rounded transition-all ${
                                task.completed
                                  ? "bg-blue-50 border border-blue-200 text-blue-900"
                                  : task.inProgress
                                  ? "bg-yellow-50 border border-yellow-200 text-yellow-900"
                                  : "bg-white border border-gray-200 text-gray-700"
                              }`}
                            >
                              <div className="flex items-start gap-2">
                                <div
                                  className={`w-1.5 h-1.5 rounded-full flex-shrink-0 mt-1 ${
                                    task.completed
                                      ? "bg-blue-500"
                                      : task.inProgress
                                      ? "bg-yellow-500"
                                      : task.priority === "Deep"
                                      ? "bg-purple-500"
                                      : task.priority === "Normal"
                                      ? "bg-blue-400"
                                      : "bg-green-500"
                                  }`}
                                />
                                <div className="flex-1 min-w-0">
                                  <div className={`truncate font-medium ${task.completed ? "line-through" : ""}`}>
                                    {task.title}
                                  </div>
                                  <div className="flex items-center gap-1 mt-0.5">
                                    {task.completed && <span className="text-[10px] font-medium text-blue-600">✓ Completed</span>}
                                    {task.inProgress && !task.completed && (
                                      <span className="text-[10px] font-medium text-yellow-600">⏱ In Progress</span>
                                    )}
                                    {!task.completed && !task.inProgress && (
                                      <span className="text-[10px] text-gray-500">
                                        {task.duration}m · {task.priority}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                          {dayTasks.length > 3 && (
                            <div className="text-xs text-gray-500 pl-3 pt-1">+{dayTasks.length - 3} more</div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {showFeedback && feedbackTask && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl p-8 max-w-md w-full shadow-2xl">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Task Completed! 🎉</h2>
            <p className="text-gray-600 mb-6">How efficient was this session?</p>

            <div className="mb-6">
              <div className="text-sm font-medium text-gray-700 mb-2">Task: {feedbackTask.title}</div>
              <div className="flex items-center gap-4 text-sm text-gray-600">
                <span>Estimated: {feedbackTask.estimatedTime}m</span>
                <span>
                  Actual: {Math.floor(elapsedTime / 60)}m {elapsedTime % 60}s
                </span>
              </div>
            </div>

            <div className="mb-6">
              <div className="text-sm font-medium text-gray-700 mb-3">Rate your efficiency:</div>
              <div className="flex justify-center gap-2">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button key={star} onClick={() => setRating(star)} className="transition-transform hover:scale-110">
                    <Star
                      className={`w-10 h-10 ${
                        star <= rating ? "fill-yellow-400 text-yellow-400" : "text-gray-300"
                      }`}
                    />
                  </button>
                ))}
              </div>
              {rating > 0 && (
                <p className="text-center text-sm text-gray-600 mt-2">
                  {rating === 5 && "🔥 Excellent focus!"}
                  {rating === 4 && "✨ Great work!"}
                  {rating === 3 && "👍 Good effort"}
                  {rating === 2 && "😊 Room for improvement"}
                  {rating === 1 && "💪 Keep trying!"}
                </p>
              )}
            </div>

            {eegConnected && (
              <div className="mb-6 p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center gap-2 text-green-700 text-sm">
                  <Activity className="w-4 h-4" />
                  <span className="font-medium">EEG data recorded for AI analysis</span>
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowFeedback(false);
                  setRating(0);
                  setFeedbackTask(null);
                }}
                className="flex-1 px-4 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 transition-all font-medium text-gray-700"
              >
                Skip
              </button>
              <button
                onClick={submitFeedback}
                disabled={rating === 0}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl hover:from-purple-600 hover:to-purple-700 transition-all font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Submit Feedback
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
