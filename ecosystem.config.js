module.exports = {
  apps: [
    {
      name: "watcher",
      script: "main_with_watcher.py",
      interpreter: "python3",
      env: {
        ENV: "production",
        NODE_ENV: "production"
      },
      env_file: ".env.prod",
      autorestart: true,
      watch: false,
      log_date_format: "YYYY-MM-DD HH:mm Z",
      error_file: "./logs/prod-error.log",
      out_file: "./logs/prod-out.log",
      time: true,
      instances: 1,
      exec_mode: "cluster"
    },
    {
      name: "watcher_mk_image",
      script: "main_with_watcher_mk_image.py",
      interpreter: "python3",
      env: {
        ENV: "production",
        NODE_ENV: "production"
      },
      env_file: ".env.prod",
      autorestart: true,
      watch: false,
      log_date_format: "YYYY-MM-DD HH:mm Z",
      error_file: "./logs/prod-error.log",
      out_file: "./logs/prod-out.log",
      time: true,
      instances: 1,
      exec_mode: "cluster"
    }
  ]
};