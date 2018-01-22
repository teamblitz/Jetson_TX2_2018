Instructions for script usage:

1. Store the executable and scripts in /home/ubuntu/robot_programs/.

2. Modify your crontab to launch the executable at reboot:

	$ crontab -e
	
	@reboot /home/ubuntu/robot_programs/cv_track_targets

3. To check if the program is running, run ps_cv.sh.

4. To terminate the program, run kill_cv.sh.
