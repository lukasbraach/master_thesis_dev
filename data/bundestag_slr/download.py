# BUNDESTAG SLR content can be found by video ID
# on https://dbtg.tv/cvid/7605085 where 7605085 is the video ID
#
# by running curl https://webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId=7605085
# all download links can be found. The JSON document has the following format:
#
# {
#     "audioUrlMono": "",
#     "audioUrlStereo": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_mp3_128kb_stereo_de_128.mp3?fdl=1",
#     "downloadUrl": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_h264_1920_1080_5000kb_baseline_de_5000.mp4?fdl=1",
#     "downloadUrlMedium": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_h264_720_400_2000kb_baseline_de_2192.mp4?fdl=1",
#     "downloadUrlLow": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_h264_512_288_514kb_baseline_de_514.mp4?fdl=1",
#     "downloadUrlSRT": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085.srt",
#     "rssRubric": "https://webtv.bundestag.de/player/macros/bttv/podcast/video/gebaerdensprache_plenarsitzungen.xml",
#     "itunesRubric": "itpc://webtv.bundestag.de/player/macros/bttv/podcast/audio/gebaerdensprache_plenarsitzungen.xml",
#     "rubricName": "Gebärdensprache-Plenarsitzungen",
#     "itunes": "itpc://webtv.bundestag.de/player/macros/bttv/podcast/video/auswahl.xml?contentIds=7605085",
#     "terms_de": "https://www.bundestag.de/resource/blob/296016/b2b8e3ed04b91bbfb235cfed975f1a69/nutzungsbedingungen_de-data.pdf",
#     "terms_en": "https://www.bundestag.de/resource/blob/296018/062266394066e7bc6a1a1a92f9d3358e/nutzungsbedingungen_en-data.pdf",
#     "shareDisabled": false,
#     "embedDisabled": false,
#     "status": {
#         "code": 1,
#         "message": "ok"
#     }
# }