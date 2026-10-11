use std::fs;
use std::io::{self, Write};
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::Command;

pub struct MachineState {
    pub date: String,
    pub power_plan: String,
    pub ac_state: String,
    pub thread_configs: String,
    pub logical_threads: String,
    pub commit: String,
    pub dirty_worktree: bool,
    on_battery: bool,
    power_unknown: bool,
}

impl MachineState {
    pub fn capture(thread_counts: &[usize]) -> Self {
        let date = powershell("(Get-Date).ToString('o')")
            .unwrap_or_else(|| "unavailable".to_string());
        let power_plan = command_output("powercfg.exe", &["/getactivescheme"])
            .unwrap_or_else(|| "unavailable".to_string());
        let (ac_state, on_battery, power_unknown) = ac_power_state();
        let commit = command_output("git.exe", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unavailable".to_string());
        let dirty_worktree = command_output("git.exe", &["status", "--porcelain"])
            .map(|status| !status.trim().is_empty())
            .unwrap_or(true);

        Self {
            date,
            power_plan: compact(&power_plan),
            ac_state,
            thread_configs: thread_counts
                .iter()
                .map(|count| format!("{count}T"))
                .collect::<Vec<_>>()
                .join("/"),
            logical_threads: std::thread::available_parallelism()
                .map(|count| count.get().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            commit: commit.trim().to_string(),
            dirty_worktree,
            on_battery,
            power_unknown,
        }
    }

    pub fn print_header(&self) {
        println!("MACHINE STATE | date={} | power_plan={} | AC/battery={} | threads={} | logical_threads={} | commit={} | worktree={}",
            self.date,
            self.power_plan,
            self.ac_state,
            self.thread_configs,
            self.logical_threads,
            self.commit,
            if self.dirty_worktree { "dirty" } else { "clean" });
        if self.on_battery {
            eprintln!("ERROR: running on battery; refusing to start a benchmark.");
        } else if self.power_unknown {
            eprintln!("WARNING: AC/battery state could not be determined; benchmark will proceed.");
        }
    }

    pub fn require_ac_power(&self) -> io::Result<()> {
        if self.on_battery || self.power_unknown {
            Err(io::Error::new(
                io::ErrorKind::Other,
                if self.on_battery {
                    "benchmark refused because the system is on battery power"
                } else {
                    "benchmark refused because AC/battery state could not be determined"
                },
            ))
        } else {
            Ok(())
        }
    }

    pub fn capture_session(thread_counts: &[usize], session: usize) -> Self {
        let state = Self::capture(thread_counts);
        println!(
            "SESSION POWER | session={} | AC/battery={} | power_plan={}",
            session, state.ac_state, state.power_plan
        );
        state
    }

    pub fn csv_session_fields(&self) -> String {
        format!("{},{}", csv_field(&self.ac_state), csv_field(&self.power_plan))
    }

    pub fn csv_preamble(&self) -> String {
        format!(
            "# date={}\n# power_plan={}\n# ac_battery={}\n# thread_configs={}\n# logical_threads={}\n# commit={}\n# worktree={}\n",
            self.date,
            self.power_plan,
            self.ac_state,
            self.thread_configs,
            self.logical_threads,
            self.commit,
            if self.dirty_worktree { "dirty" } else { "clean" }
        )
    }

    pub fn timestamp(&self) -> String {
        self.date
            .chars()
            .take(23)
            .map(|character| match character {
                ':' => '-',
                'T' => '_',
                '.' => '-',
                other => other,
            })
            .collect()
    }
}

pub struct CsvLog {
    path: PathBuf,
    contents: String,
    flushed_bytes: usize,
}

impl CsvLog {
    pub fn create(prefix: &str, machine: &MachineState, columns: &str) -> io::Result<Self> {
        fs::create_dir_all("logs")?;
        let path = PathBuf::from("logs").join(format!("{prefix}_{}.csv", machine.timestamp()));
        let mut contents = machine.csv_preamble();
        contents.push_str(columns);
        contents.push('\n');
        fs::write(&path, &contents)?;
        let flushed_bytes = contents.len();
        Ok(Self {
            path,
            contents,
            flushed_bytes,
        })
    }

    pub fn record(&mut self, row: &str) {
        self.contents.push_str(row);
        self.contents.push('\n');
    }

    pub fn record_session_state(&mut self, session: &str, state: &MachineState) {
        self.contents.push_str(&format!(
            "# SESSION {session} | AC/battery={} | power_plan={}\n",
            state.ac_state, state.power_plan
        ));
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.flushed_bytes < self.contents.len() {
            let mut file = OpenOptions::new().append(true).open(&self.path)?;
            file.write_all(self.contents[self.flushed_bytes..].as_bytes())?;
            self.flushed_bytes = self.contents.len();
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn add_header_comment(&mut self, comment: &str) -> io::Result<()> {
        self.contents.insert_str(0, &format!("# {comment}\n"));
        fs::write(&self.path, &self.contents)?;
        self.flushed_bytes = self.contents.len();
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<PathBuf> {
        self.flush()?;
        Ok(self.path)
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn powershell(script: &str) -> Option<String> {
    command_output("powershell.exe", &["-NoProfile", "-NonInteractive", "-Command", script])
}

fn ac_power_state() -> (String, bool, bool) {
    let script = r#"
$source = 'using System.Runtime.InteropServices; using System; [StructLayout(LayoutKind.Sequential)] public struct SystemPowerStatus { public byte ACLineStatus; public byte BatteryFlag; public byte BatteryLifePercent; public byte SystemStatusFlag; public int BatteryLifeTime; public int BatteryFullLifeTime; } public static class PowerStatusApi { [DllImport("kernel32.dll")] public static extern bool GetSystemPowerStatus(out SystemPowerStatus status); }'
Add-Type -TypeDefinition $source
$status = New-Object SystemPowerStatus
[void][PowerStatusApi]::GetSystemPowerStatus([ref]$status)
"$($status.ACLineStatus)|$($status.BatteryFlag)|$($status.BatteryLifePercent)"
"#;

    let Some(result) = powershell(script) else {
        return ("unknown".to_string(), false, true);
    };
    let fields: Vec<&str> = result.trim().split('|').collect();
    match fields.first().copied() {
        Some("0") => ("battery".to_string(), true, false),
        Some("1") => ("AC power".to_string(), false, false),
        Some("255") => ("unknown AC line".to_string(), false, true),
        _ => ("unknown".to_string(), false, true),
    }
}

fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}