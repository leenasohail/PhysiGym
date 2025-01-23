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

#CompletedProcess(args=['python3', 'icebreaker.py', 'false'], returncode=0, stdout=b'run icebreaker script ...\nprocessing: output/episode00000000 ...\nprocessing: output/episode00000001 ...\nprocessing: output/episode00000002 ...\nprocessing: output/episode00000003 ...\nchecking for conc drift ice ...\nchecking for cell drift ice ...\n[True, True]\nTrue\n', stdout=b'')

# load library
import os
import platform
import subprocess
import shutil

# const
s_path_physigym = os.getcwd()
s_path_physicell = '/'.join(s_path_physigym.replace('\\','/').split('/')[:-1] + ['PhysiCell'])


# function
class TestPhysigymEpisodeClassicRandom(object):
    ''' tests for the physigym episode model for reload drift. '''

    def test_physigym_episode_classic_nonrandom(self):
        os.chdir(s_path_physigym)
        o_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src=f'{s_path_physigym}/test/icebreaker.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>1<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>1<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run(['./project'], check=False, capture_output=True)
        o_result = subprocess.run(['python3', 'icebreaker.py', 'false'], check=False, capture_output=True)
        ls_result = o_result.stdout.decode('UTF8').split('\n')
        print(ls_result)
        assert(ls_result[-2] == 'False') and \
              (ls_result[-3] == '[False, False]')

    def test_physigym_episode_classic_threadrandom(self):
        os.chdir(s_path_physigym)
        o_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src=f'{s_path_physigym}/test/icebreaker.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>4<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>0<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run(['./project'], check=False, capture_output=True)
        o_result = subprocess.run(['python3', 'icebreaker.py', 'false'], check=False, capture_output=True)
        ls_result = o_result.stdout.decode('UTF8').split('\n')
        print(ls_result)
        assert(ls_result[-2] == 'True') and \
              (ls_result[-3] == '[True, True]')

    def test_physigym_episode_classic_seedrandom(self):
        os.chdir(s_path_physigym)
        o_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src=f'{s_path_physigym}/test/icebreaker.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<max_time units="min">[0-9.]*<\/max_time>/<max_time units="min">1440.0<\/max_time>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<omp_num_threads>[0-9]*<\/omp_num_threads>/<omp_num_threads>1<\/omp_num_threads>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['sed', '-ie ', r's/<random_seed>.*<\/random_seed>/<random_seed>system_clock<\/random_seed>/g', 'config/PhysiCell_settings.xml'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'classic','-j4'], check=False, capture_output=True)
        o_result = subprocess.run(['./project'], check=False, capture_output=True)
        o_result = subprocess.run(['python3', 'icebreaker.py', 'false'], check=False, capture_output=True)
        ls_result = o_result.stdout.decode('UTF8').split('\n')
        print(ls_result)
        assert(ls_result[-2] == 'True') and \
              (ls_result[-3] == '[True, True]')


class TestPhysigymEpisodeEmbeddedRandom(object):
    ''' tests for the physigym episode model for reload drift. '''

    def test_physigym_episode_embedded_nonrandom(self):
        ''' note: be hooked up to the internet to run this test successfully.'''
        os.chdir(s_path_physigym)
        o_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src=f'{s_path_physigym}/test/icebreaker.py', dst=s_path_physicell)
        shutil.copy(src='user_projects/physigym_episode/run_physigym_episode_episodes.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','run_physigym_episode_episodes.py', '--max_time', '1440.0', '--thread', '1', '--seed', '1'], check=False, capture_output=True) #! seed not set!
        o_result = subprocess.run(['python3', 'icebreaker.py', 'false'], check=False, capture_output=True)
        ls_result = o_result.stdout.decode('UTF8').split('\n')
        print(ls_result)
        assert(ls_result[-2] == 'False') and \
              (ls_result[-3] == '[False, False]')

    def test_physigym_episode_embedded_threadrandom(self):
        ''' note: be hooked up to the internet to run this test successfully.'''
        os.chdir(s_path_physigym)
        o_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src=f'{s_path_physigym}/test/icebreaker.py', dst=s_path_physicell)
        shutil.copy(src='user_projects/physigym_episode/run_physigym_episode_episodes.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','run_physigym_episode_episodes.py', '--max_time', '1440.0', '--thread', '4', '--seed', '0'], check=False, capture_output=True)
        o_result = subprocess.run(['python3', 'icebreaker.py', 'false'], check=False, capture_output=True)
        ls_result = o_result.stdout.decode('UTF8').split('\n')
        print(ls_result)
        assert(ls_result[-2] == 'True') and \
              (ls_result[-3] == '[True, True]')

    def test_physigym_episode_embedded_seedrandom(self):
        ''' note: be hooked up to the internet to run this test successfully.'''
        os.chdir(s_path_physigym)
        o_result = subprocess.run(['python3', 'install_physigym.py', 'episode', '-f'], check=False, capture_output=True)
        os.chdir(s_path_physicell)
        shutil.copy(src=f'{s_path_physigym}/test/icebreaker.py', dst=s_path_physicell)
        shutil.copy(src='user_projects/physigym_episode/run_physigym_episode_episodes.py', dst=s_path_physicell)
        shutil.rmtree('output/', ignore_errors=True)
        o_result = subprocess.run(['make', 'data-cleanup', 'clean', 'reset'], check=False, capture_output=True)
        o_result = subprocess.run(['make', 'load', 'PROJ=physigym_episode'], check=False, capture_output=True)
        o_result = subprocess.run(['make'], check=False, capture_output=True)
        o_result = subprocess.run(['python3','run_physigym_episode_episodes.py', '--max_time', '1440.0', '--thread', '1', '--seed', 'None'], check=False, capture_output=True)
        o_result = subprocess.run(['python3', 'icebreaker.py', 'false'], check=False, capture_output=True)
        ls_result = o_result.stdout.decode('UTF8').split('\n')
        print(ls_result)
        assert(ls_result[-2] == 'True') and \
              (ls_result[-3] == '[True, True]')

