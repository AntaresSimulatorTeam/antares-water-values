from proxy_stage_cost_function import ProxyStageCostFunction
from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
from proxy_exporter import Exporter, ModifyAntaresStudy, UndoAntaresModifications
from proxy_plotter import Plotter
import os, argparse, traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from proxy_bellman_trajectories import STOCK_DISCR
from proxy_stage_cost_function import THRESHOLDS
import time

ALPHA = 2

"""
Long-Term Storage Trajectories Generator for Antares Studies

This script computes optimal storage trajectories and controls for multiple study areas
based on Monte-Carlo scenarios and a cost function parameterized by alpha.
It supports exporting Bellman values, controls, trajectories, and generating a plot.
It also allows modifying Antares study input files and undoing modifications.

The processing can be run for multiple areas in parallel, and shared study files
(hydro.ini and scenariobuilder.dat) are post-processed automatically after computation.

Usage example:

python proxy_launcher.py --dir_study "/path/to/antares/study" --areas Area1 Area2 \
    --MC_years 200 --alpha 2 --actions export_trajectories \
    
python proxy_launcher.py --dir_study "/path/to/antares/study" --areas Area1 Area2 \
    --MC_years 200 --actions modify_antares_data --TS_selection 5 10 15 20 25

Arguments:
  --dir_study       (str)    : Path to the Antares study directory (required)
  --areas           (list)   : List of study areas to process, space-separated (required)
  --MC_years        (int)    : Number of Monte-Carlo years to simulate (default: 200)
  --TS_selection    (list)   : List of TS to consider when calculating Bellman values (default: =MC_years)
  --alpha           (float)  : Cost function alpha parameter (default: 2)
  --actions         (list)   : Actions to perform (required)
      Available actions include:
      - export_controls
      - export_trajectories
      - plot_trajectories
      - modify_antares_data
      - undo_modifications

Note:
- When using 'undo_modifications' as the only action, no export directory is created.
- For parallel runs on multiple areas, results are saved in a timestamped directory under the study folder/user/.

"""

class Launch:
    def __init__(self, 
                 dir_study: str, 
                 area: str,
                 MC_years: int, 
                 alpha: float,
                 TS_selection: list[int] | None,
                 global_export_dir: str | None = None,
                ):
        """
        Initialize the Launch class with study directory, area, target area, Monte-Carlo years,
        cost function parameter alpha, fictive node flag, and global export directory.
        """
        self.dir_study = dir_study
        self.name_area = area
        self.MC_years = MC_years
        self.TS_selection = TS_selection if TS_selection is not None else list(range(MC_years))
        self.alpha = alpha
        self.global_export_dir = global_export_dir

    def run(self, actions: list[str]|None=None) -> None:
        """
        Execute requested actions including generating trajectories, exporting data, plotting,
        modifying or undoing Antares study data.
        """
        if actions is not None and len(actions) == 1 and actions[0] == "undo_modifications":
            UndoAntaresModifications(self.dir_study, self.name_area).undo_all()
            return
        
        if self.global_export_dir is None:
            raise ValueError("Global export directory must be provided unless only undo_modifications action is requested.")
        
        export_dir = os.path.join(self.global_export_dir, self.name_area)
        os.makedirs(export_dir, exist_ok=True)
        steps = ["Init Proxy", "Bellman values", "Trajectories", "Setup export/modif"]
        pbar = tqdm(total=len(steps)+52*self.MC_years+(100//STOCK_DISCR+1)*len(self.TS_selection)*51+52*self.MC_years,
                    unit="step")

        self.proxy = ProxyStageCostFunction(
            dir_study=self.dir_study,
            name_area=self.name_area,
            MC_years=self.MC_years,
            alpha=self.alpha,
            pbar=pbar
        )
        pbar.update(1)
        self.bv = BellmanValuesProxy(self.proxy,pbar=pbar,TS_selection=self.TS_selection)
        pbar.update(1)
        self.trajectories = OptimalTrajectories(self.bv,pbar=pbar)
        pbar.update(1)

        self.plotter = Plotter(self.bv, self.trajectories, export_dir=export_dir)
        self.exporter = Exporter(self.proxy, self.bv, self.trajectories, export_dir=export_dir)
        self.modifier = ModifyAntaresStudy(self.bv, self.trajectories)

        if actions is None:
            actions = ["modify_antares_data"]

        for action in actions:
            if action == "export_controls":
                self.exporter.export_controls()
            elif action == "export_trajectories":
                self.exporter.export_trajectories()
            elif action == "plot_trajectories":
                self.plotter.plot_trajectories()
            elif action == "modify_antares_data":
                self.modifier.apply_all()
            elif action == "undo_modifications":
                UndoAntaresModifications(self.dir_study, self.name_area).undo_all()
            else:
                print(f"Unknown action: {action}")
        pbar.update(1)

def run_for_area(area: str,
                 dir_study: str, 
                 MC_years: int,
                 TS_selection: list[int] | None, 
                 alpha: float,
                 actions: list[str] | None = None, 
                 global_export_dir: str | None = None) -> None:
    """
    Launch the processing for a single area with given parameters and actions.
    If action is only undo_modifications, does not pass export directory.
    """
    if actions is not None and len(actions) == 1 and actions[0] == "undo_modifications":
        Launch(
            dir_study=dir_study,
            area=area,
            MC_years=MC_years,
            TS_selection=TS_selection,
            alpha=alpha,
            global_export_dir=None
        ).run(actions=actions)
    else:
        Launch(
            dir_study=dir_study,
            area=area,
            MC_years=MC_years,
            TS_selection=TS_selection,
            alpha=alpha,
            global_export_dir=global_export_dir,
        ).run(actions=actions)


def post_process_shared_files(dir_study: str, areas: list[str]) -> None:
    """
    Post-process shared study files (hydro.ini and scenariobuilder.dat)
    after all parallel computations to ensure consistency.
    """
    # Modify scenariobuilder.dat by appending lines from temporary files
    sb_lines = []
    for area in areas:
        file_path = os.path.join(dir_study,"user", "tmp", "scenariobuilder_lines", f"{area}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                sb_lines.extend(f.readlines())

    sb_path = os.path.join(dir_study, "settings", "scenariobuilder.dat")
    with open(sb_path, "a") as f:
        f.writelines(sb_lines)


def main() -> None:
    """
    Main entry point for the script.
    Parses command line arguments, runs the requested actions on specified areas,
    optionally in parallel, then post-processes shared files.
    """
    parser = argparse.ArgumentParser(description="Launch the generation of storage trajectories for multiple areas.")
    parser.add_argument("--dir_study", type=str, required=True, help="Antares study directory.")
    parser.add_argument("--areas", type=str, nargs='+', required=True, help="List of study areas (space-separated).")
    parser.add_argument("--MC_years", type=int, required=False, default=200, help="Number of Monte-Carlo years to simulate.")
    parser.add_argument("--TS_selection", type=int, nargs='+', default=None, help="List of TS to consider when calculating Bellman values. Default is all TS.")
    parser.add_argument("--actions", type=str, nargs='*', default=None, help="List of actions to perform.")


    args = parser.parse_args()

    # If only undo_modifications action, do not create export directories
    if args.actions is not None and len(args.actions) == 1 and args.actions[0] == "undo_modifications":
        for area in args.areas:
            run_for_area(
                area,
                args.dir_study,
                args.MC_years,
                args.TS_selection,
                ALPHA,
                args.actions,
                None
            )
        return

    t_start = time.perf_counter()

    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date_pretty = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    global_export_dir = os.path.join(
        args.dir_study,
        "user",
        f"LT_storage_trajectories_{run_date}"
    )
    os.makedirs(global_export_dir, exist_ok=True)

    log_path = os.path.join(global_export_dir, "run_log.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== LT storage trajectories run ===\n")
        f.write(f"Date                : {run_date_pretty}\n")
        f.write(f"dir_study           : {args.dir_study}\n")
        f.write(f"areas               : {args.areas}\n")
        f.write(f"actions             : {args.actions}\n")
        f.write(f"MC_years            : {args.MC_years}\n")
        f.write(f"TS_selection        : {args.TS_selection}\n")
        f.write(f"STOCK_DISCR         : {STOCK_DISCR}\n")
        f.write(f"THRESHOLDS          : {THRESHOLDS}\n")
        f.write(f"alpha               : {ALPHA}\n")
        f.write("\n")

    if len(args.areas) == 1:
        run_for_area(
            args.areas[0],
            args.dir_study,
            args.MC_years,
            args.TS_selection,
            ALPHA,
            args.actions,
            global_export_dir,
        )
    else:
        with ProcessPoolExecutor() as executor:
            future_to_area = {
                executor.submit(
                    run_for_area,
                    area,
                    args.dir_study,
                    args.MC_years,
                    args.TS_selection,
                    ALPHA,
                    args.actions,
                    global_export_dir
                ): area for area in args.areas
            }
            for future in as_completed(future_to_area):
                area = future_to_area[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Error for area {area}: {e}")
                    traceback.print_exc()

    post_process_shared_files(args.dir_study, args.areas)

    t_end = time.perf_counter()
    elapsed_s = t_end - t_start

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Total runtime (s)    : {elapsed_s:.2f}\n")

    print(f"📝 Run log written to {log_path}")


if __name__ == "__main__":
    main()
