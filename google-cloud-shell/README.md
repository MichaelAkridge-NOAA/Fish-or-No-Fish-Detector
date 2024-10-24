# Google Cloud Fish Detector NODD App

## Start the Google Cloud Shell 
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FMichaelAkridge-NOAA%2FFish-or-No-Fish-Detector&cloudshell_git_branch=MOUSS_2016&cloudshell_print=cloud-shell-readme.txt&cloudshell_workspace=google-cloud-shell&cloudshell_tutorial=TUTORIAL.md)


## Contact
- michael.akridge@noaa.gov

### More info on Google Cloud Shell:
- https://cloud.google.com/shell/docs/how-cloud-shell-works
- https://cloud.google.com/shell/docs/open-in-cloud-shell
- https://cloud.google.com/shell/docs/configuring-cloud-shell#environment_customization

## Build Notes
```
docker build -t gcs-fish-detector .
docker tag gcs-fish-detector michaelakridge326/gcs-fish-detector:latest
docker push michaelakridge326/gcs-fish-detector:latest
```
