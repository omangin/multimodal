#!/usr/bin/env python


"""Runner script for TwoModalitiesExperiment.
For use with expjobs.
"""


from expjobs.helpers import run_class

from multimodal.experiment import ThreeModalitiesExperiment


class NewThreeModalitiesExperiment(ThreeModalitiesExperiment):
    """Also print results...
    """

    def run(self):
        super(NewThreeModalitiesExperiment, self).run()
        self.print_result_table()


run_class(NewThreeModalitiesExperiment)
