module.exports = {
  apps: [
    {
      name: "watcher",
      script: "./main_with_watcher.py",
      interpreter: "python3",
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      watch: false,
      env_file: ".env.prod",
      log_date_format: "YYYY-MM-DD HH:mm Z",
      error_file: "./logs/prod-error.log",
      out_file: "./logs/prod-out.log",
      time: true,
      env: {
        ENV: "prod",
        NODE_ENV: "production"
      },
    },
    {
      name: "watcher_image",
      script: "./main_with_watcher_mk_image.py",
      interpreter: "python3",
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      watch: false,
      env_file: ".env.prod",
      log_date_format: "YYYY-MM-DD HH:mm Z",
      error_file: "./logs/prod-error.log",
      out_file: "./logs/prod-out.log",
      time: true,
      env: {
        ENV: "prod",
        NODE_ENV: "production"
      },
    },
  ]
};
