from proxy_stage_cost_function import Proxy
from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
from proxy_exporter import Exporter, ModifyAntaresStudy, UndoAntaresModifications
from proxy_plotter import Plotter
import time, os, argparse, traceback, ast
from datetime import datetime
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor, as_completed




class Launch:
    def __init__(self, 
                 dir_study: str, 
                 area: str,
                 area_target:str|None, 
                 MC_years: list, 
                 alpha: float, 
                 enable_logging: bool, 
                 global_export_dir: str | None = None):
        
        self.dir_study = dir_study
        self.name_area = area
        self.MC_years = MC_years
        self.alpha = alpha
        self.enable_logging = enable_logging
        self.global_export_dir = global_export_dir
        self.area_target = area_target if area_target else area  # Use area_target if provided, otherwise use area

    def run(self, actions: list[str] | None = None) -> None:
        # Si uniquement undo_modifications, ne crée aucun dossier d'export ni objet inutile
        if actions is not None and len(actions) == 1 and actions[0] == "undo_modifications":
            UndoAntaresModifications(self.dir_study, self.name_area, self.area_target).undo_all()
            return

        if self.global_export_dir is None:
            raise ValueError("A per-area export_dir must be provided via global_export_dir in all cases except undo_modifications.")
        export_dir = os.path.join(self.global_export_dir, self.name_area)
        os.makedirs(export_dir, exist_ok=True)

        start = time.time()
        self.proxy = Proxy(dir_study=self.dir_study, name_area = self.name_area, MC_years= self.MC_years, alpha=self.alpha)
        self.bv = BellmanValuesProxy(self.proxy,enable_logging=self.enable_logging, export_dir=export_dir)
        self.trajectories = OptimalTrajectories(self.bv)
        end = time.time()
        print(f"Stage cost functions, Bellman values and trajectories for area {self.name_area} computed in : {end-start} s.")

        self.plotter = Plotter(self.bv,self.trajectories)
        self.exporter = Exporter(self.proxy, self.bv,self.trajectories)
        self.modifier = ModifyAntaresStudy(self.bv,self.trajectories,self.area_target)

        if actions is None:
            actions = ["modify_antares_data"]
        if actions == ["all"]:
            actions = [
                "export_bellman_values",
                "export_controls",
                "export_trajectories",
                "plot_trajectories",
                "plot_usage_values",
                "plot_usage_values_heatmap",
                "plot_all_trajectories_pyplot",
                "plot_adjusted_rule_curves",
                "modify_antares_data",
            ]

        for action in actions:
            if action == "export_bellman_values":
                self.exporter.export_bellman_values()
            elif action == "export_controls":
                self.exporter.export_controls()
            elif action == "export_trajectories":
                self.exporter.export_trajectories()
            elif action == "plot_trajectories":
                self.plotter.plot_trajectories()
            elif action == "plot_usage_values":
                self.plotter.plot_usage_values()
            elif action == "plot_usage_values_heatmap":
                self.plotter.plot_usage_values_heatmap()
            elif action == "plot_all_trajectories_pyplot":
                self.plotter.plot_all_trajectories_pyplot()
            elif action == "plot_adjusted_rule_curves":
                self.plotter.plot_adjusted_rule_curves()
            elif action == "modify_antares_data":
                self.modifier.apply_all()
            elif action == "undo_modifications":
                UndoAntaresModifications(self.dir_study, self.name_area,self.area_target).undo_all()
            else:
                print(f"Unknown action: {action}")

def run_for_area(area: str,area_target:str|None, 
                 dir_study: str, 
                 MC_years: list, 
                 alpha: float,
                 enable_logging: bool, 
                 actions: list[str] | None = None, 
                 global_export_dir: str | None = None) -> None:
    # Si uniquement undo_modifications, ne passe pas d'export dir
    if actions is not None and len(actions) == 1 and actions[0] == "undo_modifications":
        Launch(
            dir_study=dir_study,
            area=area,
            area_target=area_target,
            MC_years=MC_years,
            alpha=alpha,
            enable_logging=enable_logging,
            global_export_dir=None,
        ).run(actions=actions)
    else:
        Launch(
            dir_study=dir_study,
            area=area,
            area_target=area_target,
            MC_years=MC_years,
            alpha=alpha,
            enable_logging=enable_logging,
            global_export_dir=global_export_dir,
        ).run(actions=actions)


def parse_mc_years(value: str) -> list:
    """
    Convertit une chaîne en liste d'indices MC (base 0).
    - Si l'utilisateur entre un entier `n`, retourne [0, ..., n-1].
    - Si l'utilisateur entre une liste [1, 5, 10], retourne [0, 4, 9].
    """
    try:
        # Entrée de type entier
        n = int(value)
        if n < 1:
            raise argparse.ArgumentTypeError("Le nombre d'années doit être >= 1.")
        return list(range(n))  # Déjà base 0
    except ValueError:
        try:
            # Entrée de type liste
            val = ast.literal_eval(value)
            if isinstance(val, list) and all(isinstance(x, int) for x in val):
                if not all(x >= 1 for x in val):
                    raise argparse.ArgumentTypeError("Tous les indices de la liste doivent être >= 1.")
                return [x - 1 for x in val]  # Décalage base 1 → base 0
        except Exception:
            pass
        raise argparse.ArgumentTypeError(
            "MC_years doit être un entier (ex: 100) ou une liste d'entiers >= 1 (ex: [1, 5, 10])."
        )

def post_process_shared_files(dir_study: str, areas: list[str], area_target:str) -> None:
    # ✅ Modifier hydro.ini
    hydro_ini_path = os.path.join(dir_study, "input", "hydro", "hydro.ini")
    config = ConfigParser()
    config.read(hydro_ini_path)

    for area in areas:
        flag_path = os.path.join(dir_study, "tmp", "hydro_flags", f"{area}.flag")
        if os.path.exists(flag_path):
            if "reservoir" not in config:
                config["reservoir"] = {}
            config["reservoir"][area] = "false"

    with open(hydro_ini_path, "w") as configfile:
        config.write(configfile)

    # ✅ Modifier scenariobuilder.dat
    sb_lines = []
    if area_target is None:
        for area in areas:
            file_path = os.path.join(dir_study, "tmp", "scenariobuilder_lines", f"{area}.txt")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    sb_lines.extend(f.readlines())
    else:
        file_path = os.path.join(dir_study, "tmp", "scenariobuilder_lines", f"{area_target}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                sb_lines.extend(f.readlines())

    sb_path = os.path.join(dir_study, "settings", "scenariobuilder.dat")
    with open(sb_path, "a") as f:
        f.writelines(sb_lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Lancer la génération des trajectoires pour plusieurs zones.")
    parser.add_argument("--dir_study", type=str, required=True, help="Répertoire de l'étude Antares.")
    parser.add_argument("--areas", type=str, nargs='+', required=True, help="Liste des zones d'étude (séparées par un espace).")
    parser.add_argument("--MC_years", type=parse_mc_years, required=False,default=list(range(200)), help="Nombre d'années de Monte Carlo ou liste d'années entre guillemets (ex: 100 ou [1,2,3].")
    parser.add_argument("--alpha", type=float, required=False,default=2, help="Coefficient alpha de la fonction de coût, par défaut vaut 2.")
    parser.add_argument("--enable_logging", type=bool, default=False, help="Activer les logs.")
    parser.add_argument("--actions", type=str, nargs='*', default=None, help="Liste des actions à effectuer")
    parser.add_argument("--area_target", type=str, required=False,default=None,help="Zone cible pour les modifications, si None utilise la zone actuelle.")

    args = parser.parse_args()

    # Si uniquement undo_modifications, ne crée aucun dossier d'export
    if args.actions is not None and len(args.actions) == 1 and args.actions[0] == "undo_modifications":
        for area in args.areas:
            run_for_area(
                area,
                args.area_target,
                args.dir_study,
                args.MC_years,
                args.alpha,
                args.enable_logging,
                args.actions,
                None
            )
        return

    # Sinon, création du dossier global d'export (daté)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_export_dir = os.path.join(args.dir_study, f"exports_LT_storage_trajectories_{date_str}")
    os.makedirs(global_export_dir, exist_ok=True)

    if len(args.areas) == 1:
        run_for_area(
            args.areas[0],
            args.area_target,
            args.dir_study,
            args.MC_years,
            args.alpha,
            args.enable_logging,
            args.actions,
            global_export_dir,
        )
    else:
        with ProcessPoolExecutor() as executor:
            future_to_area = {
                executor.submit(
                    run_for_area,
                    area,
                    args.area_target,
                    args.dir_study,
                    args.MC_years,
                    args.alpha,
                    args.enable_logging,
                    args.actions,
                    global_export_dir
                ): area for area in args.areas
            }
            for future in as_completed(future_to_area):
                area = future_to_area[future]
                try:
                    future.result()
                except Exception as e:
                    
                    print(f"❌ Erreur pour la zone {area} : {e}")
                    traceback.print_exc()

    # Post-traitement des fichiers partagés
    post_process_shared_files(args.dir_study, args.areas, args.area_target)

if __name__ == "__main__":
    main()
