###
# title: test_physigym_model.py
#
# language: python3 and command line
# author: Alexandre Bertin, Elmar Bucher
# date: 2025-01-20
# license: BSD 3-Clause
#
# description:
#   pytest unit test library for the physigym modeling framework.
#   + https://docs.pytest.org/
#
# note:
#   assert actual == expected, message
#   == value equality
#   is reference equality
#   pytest.approx for real values
#####


# load library
import os
import subprocess
import shutil

# const
s_path_physigym = os.getcwd()
s_path_physicell = "/".join(
    s_path_physigym.replace("\\", "/").split("/")[:-1] + ["PhysiCell"]
)


# function

class TestPhysigymTemplate(object):
    """ tests for the physigym template model. """

    def test_physigym_template_classic(self):
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "template", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_template'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>4<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>system_clock<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run(['./project'], check=False, capture_output=True)
        #print("\n",o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000024.xml")) and \
              (os.path.exists("output/episode00000001/output00000024.xml")) and \
              (os.path.exists("output/episode00000002/output00000024.xml")) and \
              (os.path.exists("output/episode00000003/output00000024.xml"))
        # rest output folder
        shutil.rmtree("output/")

    def test_physigym_template_embedded(self):
        """ note: be hooked up to the internet to run this test successfully. """
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "template", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_template'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','custom_modules/physigym/physigym/envs/run_physigym_template_episodes.py', '--max_time', '1440.0', '--thread', '4', '--seed', 'None'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000023.xml")) and \
              (os.path.exists("output/episode00000001/output00000023.xml")) and \
              (os.path.exists("output/episode00000002/output00000023.xml"))
        # rest output folder
        shutil.rmtree("output/")


class TestPhysigymTutorial(object):
    """ tests for the physigym tutorial model. """

    def test_physigym_tutorial_classic(self):
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "tutorial", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_tutorial'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>4<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>system_clock<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run(['./project'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000024.xml")) and \
              (os.path.exists("output/episode00000001/output00000024.xml")) and \
              (os.path.exists("output/episode00000002/output00000024.xml")) and \
              (os.path.exists("output/episode00000003/output00000024.xml"))
        # rest output folder
        shutil.rmtree("output/")

    def test_physigym_tutorial_embedded(self):
        """ note: be hooked up to the internet to run this test successfully. """
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "tutorial", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_tutorial'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','custom_modules/physigym/physigym/envs/run_physigym_tutorial_episodes.py', '--max_time', '1440.0', '--thread', '4', '--seed', 'None'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000023.xml")) and \
              (os.path.exists("output/episode00000001/output00000023.xml")) and \
              (os.path.exists("output/episode00000002/output00000023.xml"))
        # rest output folder
        shutil.rmtree("output/")


class TestPhysigymEpisode(object):
    """ tests for the physigym episode model. """

    def test_physigym_episode_classic(self):
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "episode", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>4<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>system_clock<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run(['./project'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000024.xml")) and \
              (os.path.exists("output/episode00000001/output00000024.xml")) and \
              (os.path.exists("output/episode00000002/output00000024.xml")) and \
              (os.path.exists("output/episode00000003/output00000024.xml"))
        # rest output folder
        shutil.rmtree("output/")

    def test_physigym_episode_embedded(self):
        """ note: be hooked up to the internet to run this test successfully. """
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "episode", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','custom_modules/physigym/physigym/envs/run_physigym_episode_episodes.py', '--max_time', '1440.0', '--thread', '4', '--seed', 'None'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000023.xml")) and \
              (os.path.exists("output/episode00000001/output00000023.xml")) and \
              (os.path.exists("output/episode00000002/output00000023.xml"))
        # rest output folder
        shutil.rmtree("output/")


class TestPhysigymTib(object):
    """ tests for the physigym tme model. """

    def test_physigym_tib_classic(self):
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "tumor_immune_base", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_tumor_immune_base'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>4<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>system_clock<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run([f'{s_path_physicell}/project'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000024.xml")) and \
              (os.path.exists("output/episode00000001/output00000024.xml")) and \
              (os.path.exists("output/episode00000002/output00000024.xml")) and \
              (os.path.exists("output/episode00000003/output00000024.xml"))
        # rest output folder
        shutil.rmtree("output/")


    def test_physigym_tib_embedded(self):
        """ note: be hooked up to the internet to run this test successfully. """
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "tumor_immune_base", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_tumor_immune_base'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','custom_modules/physigym/physigym/envs/run_physigym_tib_episodes.py', '--max_time', '1440.0', '--thread', '4', '--seed', 'None'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/episode00000000/output00000023.xml")) and \
              (os.path.exists("output/episode00000001/output00000023.xml")) and \
              (os.path.exists("output/episode00000002/output00000023.xml"))
        # rest output folder
        shutil.rmtree("output/")


    def test_physigym_tib_rl(self):
        """ note: be hooked up to the internet to run this test successfully. """
        s_unittest = "test_physigym_tib_rl"
        # install model
        os.chdir(s_path_physigym)
        o_result = subprocess.run(["python3", "install_physigym.py", "tumor_immune_base", "-f"], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        # rest output folder
        shutil.rmtree("output/", ignore_errors=True)
        for s_dir in os.listdir("tensorboard/"):
            if (s_dir.startswith(s_unittest)):
                shutil.rmtree(f"tensorboard/{s_dir}")
        # load, compile, and run model
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_tumor_immune_base'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>4<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>system_clock<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', rf's/name: str = ".*"/name: str = "{s_unittest}"/g', 'custom_modules/physigym/physigym/envs/sac_tib.py'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/cuda:.*bool.*=.*True/cuda: bool = False/g', 'custom_modules/physigym/physigym/envs/sac_tib.py'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/wandb_track:.*bool.*=.*True/wandb_track: bool = False/g', 'custom_modules/physigym/physigym/envs/sac_tib.py'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/total_timesteps:.*int.*=.*int(.*)/total_timesteps: int = int(72)/g', 'custom_modules/physigym/physigym/envs/sac_tib.py'], check=False, capture_output=True)

        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','custom_modules/physigym/physigym/envs/sac_tib.py'], check=False, capture_output=True)
        #print("\n", o_result)
        # test for output
        assert(os.path.exists("output/output00000023.xml")) and \
              (os.path.exists("tensorboard/")) and \
              (any([s_dir.startswith(s_unittest) for s_dir in os.listdir("tensorboard/")]))
        # reset output folder
        shutil.rmtree("output/")
        for s_dir in os.listdir("tensorboard/"):
            if (s_dir.startswith(s_unittest)):
                shutil.rmtree(f"tensorboard/{s_dir}")
