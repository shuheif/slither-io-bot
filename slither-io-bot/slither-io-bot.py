import time, datetime

from window_controller import WindowController
from path_planner import PathPlanner

def main():
    controller = WindowController()
    planner = PathPlanner('random', controller.get_snake_head())
    controller.open_browser()
    start_game(controller)
    start_timestamp = datetime.datetime.now()
    movement_counter = 0
    timestamp = start_timestamp
    while not on_main_menu(controller):
        if datetime.datetime.now() - timestamp > datetime.timedelta(milliseconds=1000):
            timestamp = datetime.datetime.now()
            movement_counter += 1
            lines, pellets, image = controller.get_local_map()
            print('{} lines and {} pellets'.format(lines.shape[0], len(pellets)))
            radian = planner.get_radian(lines, pellets)
            print('radian: {}'.format(radian))
            controller.click_radian(radian)
    calculate_game_time(start_timestamp)
    print('Total movement count: {}'.format(movement_counter))
    time.sleep(3)
    print('Last score was {}'.format(controller.read_last_score()))
    controller.destroy_all_windows()

def start_game(controller):
    while on_main_menu(controller):
        time.sleep(1)
        controller.click_play_game()
    time.sleep(1)
    controller.click_radian(0)

def on_main_menu(controller) -> bool:
    return controller.is_change_skin_visible()

def calculate_game_time(start_timestamp):
    delta = datetime.datetime.now() - start_timestamp
    print('Total game time: {}'.format(delta))

if __name__ == '__main__':
    main()
