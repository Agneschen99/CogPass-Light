This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# Neuroplan 🧠📅
EEG-based personalized learning planner with two modes:
- **Light Mode (Next.js)**: Smart study planner with task scheduling.
- **EEG Mode (Streamlit)**: Brainwave-based focus tracking and analysis (Muse device).
 
---
## 🚀 Getting Started

## Requirements
- Node.js 18+ and npm  
- Python 3.9+ (recommend 3.10/3.11) and pip

## Light Mode (Next.js)

**Quickly complete the environment setup by script**
reset-and-run-min.ps1

reset-and-run-min.cmd

Download these two files to the project root directory (at the same level as package.json).

Double click reset and run min.cmd (or execute it in PowerShell .\reset-and-run-min.cmd).

It will automatically: close residual nodes → clean up node_comodules/. next → npm install → npm run dev.

Open in browser:** http://localhost:3000 **(or the Local/Network address displayed in the terminal)

If the above methods do not work, try the following methods.

**1. Clone the repository**
```bash
git clone https://github.com/Agneschen99/Neuroplan.git
cd Neuroplan

**2. Install dependencies**
```bash
npm install

**3. Set up environment variables**
 Copy .env.example to .env.local and update with your own values:
```bash
cp .env.example .env.local
.env.local example:
MODEL_BACKEND_URL=http://localhost:8000
MODEL_TOKEN=your_token_here
NODE_ENV=development

**4. Run the development server**
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```
**Open** http://localhost:3000
 in your browser to see the app.
If port 3000 is busy, the app will run on another port (e.g., 3001).

**EEG Mode (Streamlit)**
1. Enter the EEG app folder
```bash
cd eeg_app

2. Create virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run streamlit_app.py

Open http://localhost:8501
 in your browser to view the EEG dashboard.

**Project Structure**
├── public/  # Static files

├── src/  # Light Mode (Next.js)

├── eeg_app/  # EEG Mode (Streamlit)

├── .env.example # Example environment file

├── .gitignore # Ignored files

├── package.json # Dependencies

└── README.md # Project documentation

**Commit Convention**
We follow Conventional Commits:
1. feat: add a new feature
2. fix: bug fix
3. docs: documentation only changes
4. style: formatting, no code logic changes
5. refactor: code change that neither fixes a bug nor adds a feature
6. test: add/modify tests
7. chore: build process or auxiliary tools changes
   
**Example:**

feat: add EEG mode wire-up

fix: resolve calendar rendering bug


You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
