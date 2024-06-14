class State:
    def __init__(self, game):
        self.game = game

    def enter(self):
        pass

    def exit(self):
        pass

    def update(self, delta_time):
        pass

    def render(self):
        pass

    def handle_event(self, event):
        pass

class MainMenuState(State):
    def enter(self):
        print("Entering Main Menu")

    def update(self, delta_time):
        pass

    def render(self):
        print("Rendering Main Menu")

    def handle_event(self, event):
        if event.type == 'KEYDOWN' and event.key == 'ENTER':
            self.game.state_manager.change_state('GameState')

class GameState(State):
    def enter(self):
        print("Entering Game State")

    def update(self, delta_time):
        self.game.update()

    def render(self):
        self.game.render()

class StateManager:
    def __init__(self, game):
        self.game = game
        self.states = {}
        self.active_state = None

    def add_state(self, state_name, state):
        self.states[state_name] = state

    def change_state(self, new_state_name):
        if self.active_state:
            self.active_state.exit()
        self.active_state = self.states[new_state_name]
        self.active_state.enter()

    def update(self, delta_time):
        if self.active_state:
            self.active_state.update(delta_time)

    def render(self):
        if self.active_state:
            self.active_state.render()

    def handle_event(self, event):
        if self.active_state:
            self.active_state.handle_event(event)
