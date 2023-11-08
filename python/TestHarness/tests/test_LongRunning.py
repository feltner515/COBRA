#* This file is part of the MOOSE framework
#* https://www.mooseframework.org
#*
#* All rights reserved, see COPYRIGHT for full restrictions
#* https://github.com/idaholab/moose/blob/master/COPYRIGHT
#*
#* Licensed under LGPL 2.1, please see LICENSE for details
#* https://www.gnu.org/licenses/lgpl-2.1.html

from TestHarnessTestCase import TestHarnessTestCase

class TestHarnessTester(TestHarnessTestCase):
    def testLongRunningStatus(self):
        """
        Test for RUNNING status in the TestHarness
        """
        output = self.runTests('-i', 'long_running').decode('utf-8')
        self.assertIn('RUNNING', output)
        self.assertIn('[FINISHED]', output)
