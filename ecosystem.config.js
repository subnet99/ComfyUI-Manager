const path = require('path');

const restartCommand = process.env.RESTART_COMMAND || '';
let args = '--resource-url https://pub-79bc862635254c60af6fca612486fdb9.r2.dev/install.json --interval 120';
if (restartCommand) {
  args += ` --restart-command "${restartCommand}"`;
}

module.exports = {
  apps: [{
    name: 'comfyui-installer',
    script: path.join(__dirname, 'direct_installer.py'),
    args: args,
    interpreter: path.join(__dirname, '..', '..', '.venv', 'bin', 'python'),
    cwd: path.join(__dirname, '..', '..'),
    env: {
      COMFYUI_PATH: path.join(__dirname, '..', '..'),
      PYTHON_EXECUTABLE: path.join(__dirname, '..', '..', '.venv', 'bin', 'python'),
      NODE_ENV: 'production',
      RESTART_COMMAND: restartCommand
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    time: true,
    min_uptime: '10s',
    max_restarts: 10,
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    kill_timeout: 5000,
    wait_ready: true,
    listen_timeout: 10000
  }]
};
