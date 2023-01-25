import ast
import json
import os
import pandas as pd
import time

import src.Calls as Calls
import src.util as util
from typing import Dict

from src.datatypes import Library
from src.pipeline_abstraction import NodeVisitor
from src.datatypes import GraphInformation
from src.json_to_rdf import build_pipeline_rdf_page, build_default_rdf_page, build_library_rdf_page

libraries: Dict[str, Library]
libraries = dict()
default_graph = []

target_files = [
    'sammy123.lower-back-pain-symptoms-dataset',
    'thomaskonstantin.top-270-rated-computer-science-programing-books',
    'yersever.500-person-gender-height-weight-bodymassindex',
    'neuromusic.avocado-prices',
    'fabdelja.autism-screening-for-toddlers',
    'jpacse.datasets-for-churn-telecom',
    'vedavyasv.usa-housing',
    'mrmorj.dataset-of-songs-in-spotify',
    'promptcloud.jobs-on-naukricom',
    'wduckett.moneyball-mlb-stats-19622012',
    'mylesoneill.pokemon-sun-and-moon-gen-7-stats',
    'tejashvi14.medical-insurance-premium-prediction',
    'mabusalah.brent-oil-prices',
    'ruchi798.source-based-news-classification',
    'mruanova.us-gasoline-and-diesel-retail-prices-19952021',
    'prathamtripathi.drug-classification',
    'yasserh.wine-quality-dataset',
    'deepu1109.star-dataset',
    'mit.pantheon-project',
    'fedesoriano.hepatitis-c-dataset',
    'nicholasjhana.energy-consumption-generation-prices-and-weather',
    'anmolkumar.health-insurance-cross-sell-prediction',
    'marklvl.bike-sharing-dataset',
    'uciml.forest-cover-type-dataset',
    'fizzbuzz.cleaned-toxic-comments',
    'mrisdal.married-at-first-sight',
    'christianlillelund.passenger-list-for-the-estonia-ferry-disaster',
    'hsankesara.medium-articles',
    'itachi9604.disease-symptom-description-dataset',
    'mdabbert.ultimate-ufc-dataset',
    'camnugent.california-housing-prices',
    'cityofLA.la-restaurant-market-health-data',
    'dragonheir.logistic-regression',
    'lepchenkov.usedcarscatalog',
    'sampadab17.network-intrusion-detection',
    'andrewmvd.data-scientist-jobs',
    'noordeen.insurance-premium-prediction',
    'brittabettendorf.berlin-airbnb-data',
    'insiyeah.musicfeatures',
    'sakshigoyal7.credit-card-customers',
    'imohtn.video-games-rating-by-esrb',
    'jsphyg.tipping',
    'abcsds.pokemon',
    'shubh0799.churn-modelling',
    'giripujar.hr-analytics',
    'christianlillelund.csgo-round-winner-classification',
    'bobbyscience.league-of-legends-diamond-ranked-games-10-min',
    'benroshan.factors-affecting-campus-placement',
    'CooperUnion.cardataset',
    'siddharthm1698.coursera-course-dataset',
    'parulpandey.palmer-archipelago-antarctica-penguin-data',
    'burakhmmtgl.energy-molecule',
    'hmavrodiev.london-bike-sharing-dataset',
    'arushchillar.disneyland-reviews',
    'andrewmvd.udemy-courses',
    'crisparada.brazilian-cities',
    'vetrirah.customer',
    'arashnic.covid19-case-surveillance-public-use-dataset',
    'debajyotipodder.co2-emission-by-vehicles',
    'harrywang.wine-dataset-for-clustering',
    'grosvenpaul.family-income-and-expenditure',
    'kaushil268.disease-prediction-using-machine-learning',
    'theforcecoder.wind-power-forecasting',
    'andrewmvd.fetal-health-classification',
    'sjleshrac.airlines-customer-satisfaction',
    'rashikrahmanpritom.heart-attack-analysis-prediction-dataset',
    'spittman1248.cdc-data-nutrition-physical-activity-obesity',
    'isaikumar.creditcardfraud',
    'blastchar.telco-customer-churn',
    'rtatman.did-it-rain-in-seattle-19482017',
    'loveall.appliances-energy-prediction',
    'tanmoyx.covid19-patient-precondition-dataset',
    'filippoo.deep-learning-az-ann',
    'volodymyrgavrysh.heart-disease',
    'shivamb.real-or-fake-fake-jobposting-prediction',
    'geomack.spotifyclassification',
    'rajeevw.ufcdata',
    'nehalbirla.vehicle-dataset-from-cardekho',
    'sudalairajkumar.cryptocurrencypricehistory',
    'apratim87.housingdata',
    'nphantawee.pump-sensor-data',
    'loveall.clicks-conversion-tracking',
    'kritikseth.us-airbnb-open-data',
    'uciml.restaurant-data-with-consumer-ratings',
    'team-ai.spam-text-message-classification',
    'nareshbhat.health-care-data-set-on-heart-attack-possibility',
    'kandij.diabetes-dataset',
    'justinas.nba-players-data',
    'amitabhajoy.bengaluru-house-price-data',
    'bls.eating-health-module-dataset',
    'kukuroo3.body-performance-data',
    'adityakadiwal.water-potability',
    'birdy654.eeg-brainwave-dataset-feeling-emotions',
    'namanmanchanda.entrepreneurial-competency-in-university-students',
    'uciml.student-alcohol-consumption',
    'rtatman.chocolate-bar-ratings',
    'vjchoudhary7.hr-analytics-case-study',
    'goyalshalini93.car-data',
    'brendan45774.test-file',
    'marcospessotto.happiness-and-alcohol-consumption',
    'mirichoi0218.insurance',
    'colearninglounge.predicting-pulsar-starintermediate',
    'mpwolke.cusersmarildownloadsscarcitycsv',
    'jrobischon.wikipedia-movie-plots',
    'mboaglio.simplifiedhuarus',
    'shivam2503.diamonds',
    'mahirahmzh.starbucks-customer-retention-malaysia-survey',
    'tejashvi14.employee-future-prediction',
    'kunal28chaturvedi.covid19-and-its-impact-on-students',
    'yamqwe.depression-anxiety-stress-scales',
    'datamunge.sign-language-mnist',
    'alexteboul.diabetes-health-indicators-dataset',
    'patelprashant.employee-attrition',
    'ruslankl.mice-protein-expression',
    'adityadesai13.used-car-dataset-ford-and-mercedes',
    'zaurbegiev.my-dataset',
    'lucidlenn.sloan-digital-sky-survey',
    'imnikhilanand.heart-attack-prediction',
    'jboysen.mri-and-alzheimers',
    'shwetabh123.mall-customers',
    'uciml.breast-cancer-wisconsin-data',
    'shashwatwork.impact-of-covid19-pandemic-on-the-global-economy',
    'rakeshrau.social-network-ads',
    'shahir.protein-data-set',
    'atharvaingle.crop-recommendation-dataset',
    'elikplim.forest-fires-data-set',
    'hellbuoy.car-price-prediction',
    'arashnic.hr-analytics-job-change-of-data-scientists',
    'altruistdelhite04.loan-prediction-problem-dataset',
    'vkrahul.twitter-hate-speech',
    'ronitf.heart-disease-uci',
    'cosmos98.twitter-and-reddit-sentimental-analysis-dataset',
    'jessemostipak.hotel-booking-demand',
    'agirlcoding.all-space-missions-from-1957',
    'vjchoudhary7.customer-segmentation-tutorial-in-python',
    'nehaprabhavalkar.av-healthcare-analytics-ii',
    'jealousleopard.goodreadsbooks',
    'kondla.carinsurance',
    'zhijinzhai.loandata',
    'mnassrib.telecom-churn-datasets',
    'steveahn.memory-test-on-drugged-islanders-data',
    'abhinav89.telecom-customer',
    'rohitsahoo.sales-forecasting',
    'mazharkarimi.heart-disease-and-stroke-prevention',
    'ealaxi.banksim1',
    'fedesoriano.company-bankruptcy-prediction',
    'jmmvutu.summer-products-and-sales-in-ecommerce-wish',
    'airbnb.boston',
    'rpaguirre.tesla-stock-price',
    'navinmundhra.daily-power-generation-in-india-20172020',
    'anmolkumar.house-price-prediction-challenge',
    'secareanualin.football-events',
    'ashishpatel26.facial-expression-recognitionferchallenge',
    'terminus7.pokemon-challenge',
    'mhdzahier.travel-insurance',
    'buntyshah.auto-insurance-claims-data',
    'chirin.africa-economic-banking-and-systemic-crisis-data',
    'dipam7.student-grade-prediction',
    'fedesoriano.stroke-prediction-dataset',
    'unsdsn.world-happiness',
    'carlolepelaars.toy-dataset',
    'jenny18.honey-bee-annotated-images',
    'codersree.mount-rainier-weather-and-climbing-data',
    'barelydedicated.bank-customer-churn-modeling',
    'vikrishnan.boston-house-prices',
    'arshid.iris-flower-dataset',
    'blurredmachine.are-your-employees-burning-out',
    'cdc.national-health-and-nutrition-examination-survey',
    'volodymyrgavrysh.bank-marketing-campaigns-dataset',
    'zaraavagyan.weathercsv',
    'primaryobjects.voicegender',
    'mariaren.covid19-healthy-diet-dataset',
    'brsdincer.star-type-classification',
    'mylesoneill.world-university-rankings',
    'footprintnetwork.ecological-footprint',
    'kevinarvai.clinvar-conflicting',
    'CooperUnion.anime-recommendations-database',
    'uciml.news-aggregator-dataset',
    'kannanaikkal.food-demand-forecasting',
    'futurecorporation.epitope-prediction',
    'fedesoriano.the-boston-houseprice-data',
    'shivachandel.kc-house-data',
    'timoboz.tesla-stock-data-from-2010-to-2020',
    'alexteboul.heart-disease-health-indicators-dataset',
    'rohanrao.nifty50-stock-market-data',
    'mathchi.online-retail-data-set-from-ml-repository',
    'adammaus.predicting-churn-for-bank-customers',
    'manjeetsingh.retaildataset',
    'joshmcadams.oranges-vs-grapefruit',
    'divyansh22.us-border-crossing-data',
    'schirmerchad.bostonhoustingmlnd',
    'sindraanthony9985.marketing-data-for-a-supermarket-in-united-states',
    'iamhungundji.covid19-symptoms-checker',
    'slehkyi.extended-football-stats-for-european-leagues-xg',
    'aparnashastry.building-permit-applications-data',
    'jmmvutu.ecommerce-users-of-a-french-c2c-fashion-store',
    'santoshd3.bank-customers',
    'mansoordaku.ckdisease',
    'shrutimehta.nasa-asteroids-classification',
    'zhangjuefei.birds-bones-and-living-habits',
    'cityofLA.los-angeles-metro-bike-share-trip-data',
    'mathchi.diabetes-data-set',
    'iamsumat.spotify-top-2000s-mega-dataset',
    'brijbhushannanda1979.bigmart-sales-data',
    'brunotly.foreign-exchange-rates-per-dollar-20002019',
    'gdaley.hkracing',
    'andrewmvd.heart-failure-clinical-data',
    'nipunarora8.age-gender-and-ethnicity-face-data-csv',
    'divyansh22.flight-delay-prediction',
    'mamtadhaker.lt-vehicle-loan-default-prediction',
    'team-ai.bitcoin-price-prediction',
    'itssuru.loan-data',
    'nasa.asteroid-impacts',
    'kabure.german-credit-data-with-risk',
    'sashankpillai.spotify-top-200-charts-20202021',
    'datatattle.covid-19-nlp-text-classification',
    'lodetomasi1995.income-classification',
    'corrieaar.apartment-rental-offers-in-germany',
    'arjunbhasin2013.ccdata',
    'mateusdmachado.csgo-professional-matches',
    'kyr7plus.emg-4',
    'saurabh00007.iriscsv',
    'shivamb.5minute-crafts-video-views-dataset',
    'uciml.caravan-insurance-challenge',
    'madislemsalu.facebook-ad-campaign',
    'andrewmvd.divorce-prediction',
    'muthuj7.weather-dataset',
    'ajay1735.hmeq-data',
    'new-york-state.nys-environmental-remediation-sites',
    'sootersaalu.amazon-top-50-bestselling-books-2009-2019',
    'uciml.mushroom-classification',
    'sudalairajkumar.daily-temperature-of-major-cities',
    'garystafford.environmental-sensor-data-132k',
    'sansuthi.world-population-by-year',
    'muhakabartay.sloan-digital-sky-survey-dr16',
    'hb20007.gender-classification',
    'tejashvi14.engineering-placements-prediction',
    'nikhileswarkomati.suicide-watch',
    'tevecsystems.retail-sales-forecasting',
    'ninzaami.loan-predication',
    'elikplim.car-evaluation-data-set',
    'annavictoria.speed-dating-experiment',
    'carolzhangdc.imdb-5000-movie-dataset',
    'smid80.weatherww2',
    'berkerisen.wind-turbine-scada-dataset',
    'census.business-and-industry-reports',
    'rocki37.open-university-learning-analytics-dataset',
    'mysarahmadbhat.lung-cancer',
    'pavansubhasht.ibm-hr-analytics-attrition-dataset',
    'rgupta09.world-cup-2018-tweets',
    'puxama.bostoncsv',
    'noahgift.social-power-nba',
    'mrferozi.loan-data-for-dummy-bank',
    'prakrutchauhan.indian-candidates-for-general-election-2019',
    'sebastianmantey.nba-free-throws',
    'uciml.red-wine-quality-cortez-et-al-2009',
    'lakshmi25npathi.imdb-dataset-of-50k-movie-reviews',
    'fernandol.countries-of-the-world',
    'drgilermo.nba-players-stats-20142015',
    'idoyo92.epl-stats-20192020',
    'imakash3011.customer-personality-analysis',
    'uciml.german-credit',
    'ritesaluja.bank-note-authentication-uci-data',
    'teejmahal20.airline-passenger-satisfaction',
    'shenba.time-series-datasets',
    'hesh97.titanicdataset-traincsv',
    'johndasilva.diabetes',
    'rodsaldanha.arketing-campaign',
    'uciml.faulty-steel-plates',
    'nathanlauga.nba-games',
    'arashnic.the-depression-dataset',
    'rubenssjr.brasilian-houses-to-rent',
    'subhamjain.loan-prediction-based-on-customer-behavior',
    'uciml.pima-indians-diabetes-database',
    'kwadwoofosu.predict-test-scores-of-students',
    'NUFORC.ufo-sightings',
    'kumargh.pimaindiansdiabetescsv',
    'rohan0301.unsupervised-learning-on-country-data',
    'burak3ergun.loan-data-set',
    'frtgnn.rural-residents-daily-mobile-phone-data',
    'vikasukani.diabetes-data-set',
    'vstepanenko.disaster-tweets',
    'vbmokin.ammonium-prediction-in-river-water',
    'fedesoriano.stellar-classification-dataset-sdss17',
    'harshitshankhdhar.imdb-dataset-of-top-1000-movies-and-tv-shows',
    'jolasa.waves-measuring-buoys-data-mooloolaba',
    'uciml.indian-liver-patient-records',
    'mauryashubham.english-premier-league-players-dataset',
    'uciml.glass',
    'uciml.human-activity-recognition-with-smartphones',
    'prachi13.customer-analytics',
    'gsutters.the-human-freedom-index',
    'devashish0507.big-mart-sales-prediction',
    'brycecf.give-me-some-credit-dataset',
    'saurav9786.cardiogoodfitness',
    'jcyzag.covid19-lockdown-dates-by-country',
    'sanjeetsinghnaik.top-1000-highest-grossing-movies',
    'uciml.horse-colic',
    'fedesoriano.heart-failure-prediction',
    'arslanali4343.real-estate-dataset',
    'arpitjain007.game-of-deep-learning-ship-datasets',
    'justinas.housing-in-london',
    'paresh2047.uci-semcom',
    'tejashvi14.travel-insurance-prediction-data',
    'sohier.calcofi',
    'dronio.SolarEnergy',
    'henriqueyamahata.bank-marketing',
    'free4ever1.instagram-fake-spammer-genuine-accounts',
    'harlfoxem.housesalesprediction',
    'datafiniti.grammar-and-online-product-reviews',
    'fmejia21.nba-all-star-game-20002016',
    'dileep070.heart-disease-prediction-using-logistic-regression',
    'mathchi.churn-for-bank-customers',
    'quantbruce.real-estate-price-prediction',
    'miroslavsabo.young-people-survey',
    'shrutimechlearn.churn-modelling',
    'venky73.spam-mails-dataset',
    'shivamb.machine-predictive-maintenance-classification',
    'arslanali4343.top-personality-dataset',
    'knightbearr.pizza-price-prediction',
    'mustafaali96.weight-height',
    'datasnaek.mbti-type',
    'upadorprofzs.credit-risk',
    'cnic92.200-financial-indicators-of-us-stocks-20142018',
    'ahmettezcantekin.beginner-datasets',
    'yamqwe.omicron-covid19-variant-daily-cases',
    'spscientist.students-performance-in-exams',
    'jsphyg.weather-dataset-rattle-package',
    'shivan118.hranalysis',
    'giovamata.airlinedelaycauses',
    'gkhan496.covid19-in-turkey',
    'nehaprabhavalkar.indian-food-101',
    'samdeeplearning.deepnlp',
    'noaa.hurricane-database',
    'fedesoriano.cern-electron-collision-data',
    'ishaanv.ISLR-Auto',
    'becksddf.churn-in-telecoms-dataset',
    'sulianova.cardiovascular-disease-dataset',
    'tourist55.clothessizeprediction',
    'prakharrathi25.banking-dataset-marketing-targets',
    'shrutibhargava94.india-air-quality-data',
    'michau96.restaurant-business-rankings-2020',
    'cgurkan.airplane-crash-data-since-1908',
    'stefanoleone992.fifa-21-complete-player-dataset',
    'gregorut.videogamesales',
    'maajdl.yeh-concret-data',
    'nickhould.craft-cans',
    'jackogozaly.data-science-and-stem-salaries',
    'janiobachmann.bank-marketing-dataset',
    'doaaalsenani.usa-cers-dataset',
    'rajyellow46.wine-quality',
    'jorgesandoval.wind-power-generation',
]


def main():
    overall_start = time.time()

    OUTPUT_PATH = '../../../storage/knowledge_graph/pipelines_and_libraries/'

    dataset: os.DirEntry
    # loop through datasets & pipelines
    for dataset in os.scandir('../../../../CFGDemo/data/kaggle'):
        pipeline: os.DirEntry
        # and dataset.name in target_files
        if dataset.is_dir():
            working_file = {}
            table: os.DirEntry
            if not os.path.isdir(f'../../../../CFGDemo/kaggle/{dataset.name}'):
                continue
            for table in os.scandir(f'../../../../CFGDemo/kaggle/{dataset.name}'):
                if table.name != '.DS_Store':
                    try:
                        working_file[table.name] = pd.read_csv(table.path, nrows=1)
                    except Exception as e:
                        print("-<>", table.name, e)
            if not os.path.isdir(f'{dataset.path}/notebooks/'):
                continue
            for pipeline in os.scandir(f'{dataset.path}/notebooks'):
                if pipeline.is_dir():
                    try:
                        with open(f'{pipeline.path}/pipeline_info.json', 'r') as f:
                            pipeline_info = json.load(f)
                        with open(f'{pipeline.path}/kernel-metadata.json', 'r') as f:
                            metadata = json.load(f)
                        pipeline_info['tags'] = metadata['keywords']
                        file: os.DirEntry
                        for file in os.scandir(pipeline.path):
                            if '.py' in file.name:
                                # if metadata['id'].replace('/', '.') in target_files:
                                pipeline_analysis(working_file=working_file,
                                                  dataset=dataset,
                                                  file_path=file.path,
                                                  pipeline_info=pipeline_info,
                                                  output_path=OUTPUT_PATH,
                                                  output_filename=metadata['id'].replace('/', '.'))
                    except FileNotFoundError as e:
                        continue

    libs = [library.str() for library in libraries.values()]
    # with open('kaggle/library.json', 'w') as f:
    #     json.dump(libs, f)
    #
    # with open('kaggle/default.json', 'w') as f:
    #     json.dump(default_graph, f)

    with open(os.path.join(OUTPUT_PATH, 'library.ttl'), 'w') as f:
        f.write(build_library_rdf_page(libs))

    with open(os.path.join(OUTPUT_PATH, 'default.ttl'), 'w') as f:
        f.write(build_default_rdf_page(default_graph))

    # literals = {}
    # rdf_file = ''
    #
    # for node, value in literals.items():
    #     rdf_file = rdf_file.replace(node, value)
    #
    # with open(f'../kaggle_rdf/default.ttl', 'w') as output_file:
    #     output_file.write(rdf_file)

    overall_end = time.time()
    print(overall_end - overall_start)
    print(Calls.read_csv_call.count)


def pipeline_analysis(working_file, dataset: os.DirEntry, file_path, pipeline_info, output_path, output_filename):
    starting_time = time.time()
    SOURCE = 'kaggle'
    DATASET_NAME = dataset.name
    PYTHON_FILE_NAME = output_filename

    # Read pipeline file
    with open(file_path, 'r') as src_file:
        src = src_file.read()

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        with open(f'./errors.csv', 'a') as output_file:
            output_file.write(f'{PYTHON_FILE_NAME},{e.msg}\n')
        return

    # Initialize graph information linked list and Node Visitor
    graph = GraphInformation(PYTHON_FILE_NAME, SOURCE, DATASET_NAME, libraries)
    node_visitor = NodeVisitor(graph_information=graph)
    node_visitor.working_file = working_file

    # Pipeline analysis
    node_visitor.visit(tree)
    ending_time = time.time()
    print("Processing time: ", ending_time - starting_time, 'seconds')

    # Datastructures preparation for insertion to Neo4j
    file_elements = [el.str() for el in graph.files.values()]
    nodes = []
    head = graph.head
    line = 1
    while head is not None:
        head.generate_uri(SOURCE, DATASET_NAME, PYTHON_FILE_NAME, line)
        line += 1
        head = head.next
    head = graph.head
    while head is not None:
        nodes.append(head.str())
        head = head.next

    # with open(f'kaggle/{output_filename}.json', 'w') as f:
    #     json.dump(nodes, f)
    #
    # with open(f'kaggle/{output_filename}-files.json', 'w') as f:
    #     json.dump(file_elements, f)
    if not os.path.isdir(os.path.join(output_path, DATASET_NAME)):
        os.mkdir(os.path.join(output_path, DATASET_NAME))

    with open(os.path.join(output_path, DATASET_NAME, output_filename + '.ttl'), 'w') as f:
        f.write(build_pipeline_rdf_page(nodes, file_elements))

    pipeline_info['uri'] = util.create_pipeline_uri(SOURCE, DATASET_NAME, output_filename)
    pipeline_info['dataset'] = util.create_dataset_uri(SOURCE, DATASET_NAME)
    default_graph.append(pipeline_info)


if __name__ == "__main__":
    main()
