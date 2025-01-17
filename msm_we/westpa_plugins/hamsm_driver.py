"""Plugin for automated haMSM construction."""
import westpa
from westpa.core import extloader
from msm_we import msm_we
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


class HAMSMDriver:
    """
    WESTPA plugin to construct an haMSM.


    Can be used by including the following entries in your west.cfg::

        west:
            plugins:
            # - plugin: An augmentation plugin is also required, such as
            #           msm_we.westpa_plugins.augmentation_driver.MDAugmentationDriver
            - plugin: msm_we.westpa_plugins.hamsm_driver.HAMSMDriver
                  model_name: Name for the model
                  n_clusters: Number of clusters to place in each WE bin (see stratified clustering for more details)
                  tau: WESTPA resampling time in physical units
                  basis_pcoord_bounds: [[pcoord dim 0 lower bound, upper bound], [pcoord dim 1 lower, upper], ...]
                  target_pcoord_bounds: [[pcoord dim 0 lower bound, upper bound], [pcoord dim 1 lower, upper], ...]
                  dim_reduce_method: A string specifying a dimensionality reduction method for
                    :meth:`msm_we.msm_we.modelWE.dimReduce`
                  featurization: An importable python method implementing a featurization
                    for :meth:`msm_we.msm_we.modelWE.processCoordinates`
                  n_cpus: Number of CPUs to use with Ray
    """

    def __init__(self, sim_manager, plugin_config):

        westpa.rc.pstatus("Initializing haMSM plugin")

        assert (
            _openmp_effective_n_threads() == 1
        ), "Set $OMP_NUM_THREADS=1 for proper msm-we functionality"

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager

        self.plugin_config = plugin_config

        # Big number is low priority -- this should run after augmentation, but before other things
        self.priority = plugin_config.get("priority", 2)

        sim_manager.register_callback(
            sim_manager.finalize_run, self.construct_hamsm, self.priority
        )

        # This is defined as a class variable in the constructor, so it can be overridden if desired.
        #   By default, it's just the H5 file generated by the current WE run. But you can extend the list
        #   (for example, with
        #   h5 files from other replicates, as in the restarting plugin) before constructing the haMSM, to build a model
        #   with data from multiple runs.
        self.h5file_paths = [self.data_manager.we_h5filename]

        self.dimreduce_use_weights = self.plugin_config.get(
            "dimreduce_use_weights", True
        )

        self.dimreduce_var_cutoff = self.plugin_config.get(
            "dimreduce_var_cutoff", None
        )

        self.cross_validation_groups = self.plugin_config.get(
            "cross_validation_groups", 2
        )

        self.ray_address = self.plugin_config.get(
            "ray_address", None
        )

        self.ray_kwargs = self.plugin_config.get(
            "ray_kwargs", {}
        )

    def construct_hamsm(self):
        """
        Build an haMSM, for use with later plugins. The final constructed haMSM is stored on the data manager.
        """

        self.data_manager.hamsm_model = None

        # TODO: refPDBfile should no longer be necessary (or used anywhere) and should be safe to remove
        refPDBfile = self.plugin_config.get("ref_pdb_file")
        model_name = self.plugin_config.get("model_name")
        clusters_per_stratum = self.plugin_config.get("n_clusters")

        target_pcoord_bounds = self.plugin_config.get("target_pcoord_bounds")
        basis_pcoord_bounds = self.plugin_config.get("basis_pcoord_bounds")

        dimreduce_method = self.plugin_config.get("dimreduce_method", None)
        tau = self.plugin_config.get("tau", None)

        featurization_module = self.plugin_config.get("featurization")
        featurizer = extloader.get_object(featurization_module)
        msm_we.modelWE.processCoordinates = featurizer
        self.data_manager.processCoordinates = featurizer

        self.data_manager.close_backing()

        ray_kwargs = {"num_cpus": self.plugin_config.get("num_cpus", None)}
        ray_kwargs.update(self.ray_kwargs)

        if self.ray_address is not None:
            ray_kwargs.update({'address':self.ray_address})

        model = msm_we.modelWE()
        model.build_analyze_model(
            file_paths=self.h5file_paths,
            ref_struct=refPDBfile,
            modelName=model_name,
            basis_pcoord_bounds=basis_pcoord_bounds,
            target_pcoord_bounds=target_pcoord_bounds,
            dimreduce_method=dimreduce_method,
            n_clusters=clusters_per_stratum,
            tau=tau,
            ray_kwargs=ray_kwargs,
            step_kwargs={"dimReduce": {"use_weights": self.dimreduce_use_weights, "variance_cutoff": self.dimreduce_var_cutoff}},
            # For some reason if I don't specify fluxmatrix_iters, after the first time around
            # it'll keep using the arguments from the first time...
            # That's really alarming?
            fluxmatrix_iters=[1, -1],
            allow_validation_failure=True,  # Don't fail if a validation model fails
            cross_validation_groups=self.cross_validation_groups,
        )

        westpa.rc.pstatus(f"Storing built haMSM on {self.data_manager}")
        self.data_manager.hamsm_model = model

        return model
