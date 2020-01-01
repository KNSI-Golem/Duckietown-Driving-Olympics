def train(env, agent, config):
    obs = env.reset()
    trainer = config.build_trainer(agent)
    trainer.register_obs(obs)

    for _ in range(config.steps):
        act = trainer.model_step(obs)
        obs, done = env.step(act)

        trainer.register_obs(obs, done)

        if done:
            obs = env.reset()
            trainer.register_obs(obs)
