# Running vLLama on Windows

For Windows, the simplest way to have the vLLama script run automatically is by using the **Windows Task Scheduler**. This avoids the complexity of creating a true Windows Service.

Here are the steps to get it running:

### 1. Install Prerequisites

*   **Python:** Install Python for Windows from the [official website](https://www.python.org/downloads/windows/). Make sure to check the box that says **"Add Python to PATH"** during installation.
*   **Git:** Install Git for Windows from the [official website](https://git-scm.com/download/win).

### 2. Clone the Repository

Open a Command Prompt (`cmd.exe`) and run the following command to clone the repository to a `C:\vllama` directory:

```cmd
git clone https://github.com/erkkimon/vllama.git C:\vllama
```

### 3. Set up the Virtual Environment

Now, set up a Python virtual environment and install the required dependencies.

```cmd
cd C:\vllama
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### 4. Create a Runner Script

To make it easy for the Task Scheduler to run the script, create a new file named `run.bat` inside the `C:\vllama` directory with the following content:

```bat
@echo off
cd C:\vllama
call venv\Scripts\activate.bat
python vllama.py
```

### 5. Create a Scheduled Task

Finally, create a scheduled task to run the script automatically when your computer starts.

1.  Press `Win + R`, type `taskschd.msc`, and press Enter to open the Task Scheduler.
2.  In the right-hand pane, click **"Create Basic Task..."**
3.  **Name:** Give it a descriptive name like "vLLama Service".
4.  **Trigger:** Select **"When the computer starts"**.
5.  **Action:** Select **"Start a program"**.
6.  **Program/script:** Click **"Browse..."** and select the `C:\vllama\run.bat` file you created.
7.  Click **"Finish"** to create the task.

To ensure the script has the necessary permissions, find the task in the library, right-click it, select **Properties**, and check the box **"Run with highest privileges"**.

The script will now start automatically every time the computer boots up.
