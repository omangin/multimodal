#!/usr/bin/env python


"""Runner script for TwoModalitiesExperiment.
For use with expjobs.
"""


from expjobs.helpers import run_class

from multimodal.experiment import TwoModalitiesExperiment


class NewTwoModalitiesExperiment(TwoModalitiesExperiment):
    """Also print results...
    """

    def run(self):
        super(NewTwoModalitiesExperiment, self).run()
        self.print_result_table()


run_class(NewTwoModalitiesExperiment)
