mkdir -p data
cd data
mkdir TPAMI20
cd TPAMI20
wget https://rpg.ifi.uzh.ch/data/E2VID/datasets/TPAMI/events/hdr_selfie.zip
wget https://rpg.ifi.uzh.ch/data/E2VID/datasets/TPAMI/events/hdr_tunnel.zip
wget https://rpg.ifi.uzh.ch/data/E2VID/datasets/TPAMI/events/hdr_sun.zip
unzip hdr_selfie.zip
unzip hdr_tunnel.zip
unzip hdr_sun.zip
rm hdr_selfie.zip
rm hdr_tunnel.zip
rm hdr_sun.zip