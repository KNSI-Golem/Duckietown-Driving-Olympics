from golem_driving.agents.manual_agent import ManualAgent

agents = {
    'manual': ManualAgent
}


def get_agent(agent):
    return agents[agent]()
