from tap import Tap


class CreateDatasetArgParser(Tap):
    load_path: str = "data/data_graph"
    df_name: str = "dataset"
    self_loop: bool = False  # Compute inv. proba. with self-loop
    power_transfer: float = (  # w_min value (e.g. 1e-5) for power transfer, if passed
        None
    )
    save_plots: bool = False  # Save graph plots
    incremental: bool = True  # Skip graphs for which correlation files already exist
    compile_only: bool = False  # Only compile existing correlation files into a csv

    def configure(self) -> None:
        self.add_argument("-l", "--load_path")
        self.add_argument("-n", "--df_name")
        self.add_argument("-s", "--self_loop")
        self.add_argument("-t", "--power_transfer")
        self.add_argument("-p", "--save_plots")
        self.add_argument("-i", "--incremental")
        self.add_argument("-c", "--compile_only")
