import functions
import helper_tmva
import helpers
import ROOT
import argparse
import logging
import helper_jetclustering
import helper_flavourtagger
logger = logging.getLogger("fcclogger")

parser = functions.make_def_argparser()
args = parser.parse_args()
functions.set_threads(args)

functions.add_include_file("analyses/higgs_mass_xsec/functions.h")
functions.add_include_file("analyses/higgs_mass_xsec/functions_gen.h")


# define histograms

bins_m = (250, 0, 250)
bins_p = (200, 0, 200)
bins_m_zoom = (200, 110, 130) # 100 MeV
bins_mz = (160, 50, 130)


bins_theta = (500, 0, 5)
bins_phi = (400, -4, 4)

bins_count = (100, 0, 100)
bins_pdgid = (60, -30, 30)
bins_charge = (10, -5, 5)

bins_resolution = (10000, 0.95, 1.05)
bins_resolution_1 = (20000, 0, 2)

jet_energy = (1000, 0, 100) # 100 MeV bins
dijet_m = (2000, 0, 200) # 100 MeV bins
visMass = (2000, 0, 200) # 100 MeV bins
missEnergy  = (2000, 0, 200) # 100 MeV bins

dijet_m_final = (500, 50, 100) # 100 MeV bins

bins_cos = (100, -1, 1)
bins_aco = (1000,0,1)
bins_cosThetaMiss = (10000, 0, 1)

bins_dR = (1000, 0, 10)

bins_score = (100, 0, 1)

bins_prob = (400, 0, 2)
bins_pfcand = (200, -10, 10)

#Load in the the jet clustering and the jet flavor tagging
jet4Cluster = helper_jetclustering.ExclusiveJetClusteringHelper(4, collection="ReconstructedParticles")
jet4Flavour = helper_flavourtagger.JetFlavourHelper(jet4Cluster.jets, jet4Cluster.constituents)
path = "/home/submit/swaldy/FCCAnalyzer/data/flavourtagger/fccee_flavtagging_edm4hep_wc_v2"
jet4Flavour.load(f"{path}.json", f"{path}.onnx")


def build_graph(df, dataset):

    logging.info(f"build graph {dataset.name}")
    results, cols = [], []

    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = helpers.defineCutFlowVars(df) # make the cutX=X variables
    
    # define collections
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Alias("Particle1", "Particle#1.index")
    df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
    df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")


    # muons
    df = df.Alias("Muon0", "Muon#0.index")
    df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
    df = df.Define("muons_all_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_all)") #gets momentum of all muons
    df = df.Define("muons_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(muons_all)")
    df = df.Define("muons_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons_all)")
    df = df.Define("muons_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(muons_all)")
    df = df.Define("muons_all_no", "FCCAnalyses::ReconstructedParticle::get_n(muons_all)")

    df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(25)(muons_all)")
    df = df.Define("muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons)") #
    df = df.Define("muons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(muons)")
    df = df.Define("muons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons)")
    df = df.Define("muons_q", "FCCAnalyses::ReconstructedParticle::get_charge(muons)") #return charges of input reconstructed particles
    df = df.Define("muons_no", "FCCAnalyses::ReconstructedParticle::get_n(muons)") #get size of input collection

    
    # electrons
    df = df.Alias("Electron0", "Electron#0.index")
    df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
    df = df.Define("electrons_all_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_all)")
    df = df.Define("electrons_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(electrons_all)")
    df = df.Define("electrons_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons_all)")
    df = df.Define("electrons_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(electrons_all)")
    df = df.Define("electrons_all_no", "FCCAnalyses::ReconstructedParticle::get_n(electrons_all)")

    df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(25)(electrons_all)")
    df = df.Define("electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons)")
    df = df.Define("electrons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(electrons)")
    df = df.Define("electrons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons)")
    df = df.Define("electrons_q", "FCCAnalyses::ReconstructedParticle::get_charge(electrons)")
    df = df.Define("electrons_no", "FCCAnalyses::ReconstructedParticle::get_n(electrons)")


    # lepton kinematic histograms
    results.append(df.Histo1D(("muons_all_p_cut0", "", *bins_p), "muons_all_p"))
    results.append(df.Histo1D(("muons_all_theta_cut0", "", *bins_theta), "muons_all_theta"))
    results.append(df.Histo1D(("muons_all_phi_cut0", "", *bins_phi), "muons_all_phi"))
    results.append(df.Histo1D(("muons_all_q_cut0", "", *bins_charge), "muons_all_q"))
    results.append(df.Histo1D(("muons_all_no_cut0", "", *bins_count), "muons_all_no")) #return the size of the input collection

    results.append(df.Histo1D(("electrons_all_p_cut0", "", *bins_p), "electrons_all_p"))
    results.append(df.Histo1D(("electrons_all_theta_cut0", "", *bins_theta), "electrons_all_theta"))
    results.append(df.Histo1D(("electrons_all_phi_cut0", "", *bins_phi), "electrons_all_phi"))
    results.append(df.Histo1D(("electrons_all_q_cut0", "", *bins_charge), "electrons_all_q"))
    results.append(df.Histo1D(("electrons_all_no_cut0", "", *bins_count), "electrons_all_no"))



    #########
    ### CUT 0: all events
    #########
    results.append(df.Histo1D(("cutFlow_mumu", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_ee", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_nunu", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_uu", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_dd", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_ss", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_cc", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_bb", "", *bins_count), "cut0"))
    results.append(df.Histo1D(("cutFlow_tautau", "", *bins_count), "cut0"))
    
    #########
    ### CUT 1: select Z decay product
    #########
    df = df.Define("missingEnergy_rp", "FCCAnalyses::missingEnergy(240., ReconstructedParticles)")
    df = df.Define("missingEnergy", "missingEnergy_rp[0].energy")
    results.append(df.Histo1D(("missingEnergy_nOne", "", *missEnergy), "missingEnergy"))
    
    #select_mumu = "muons_no == 2 && electrons_no == 0 && missingEnergy < 30"
    #select_ee   = "muons_no == 0 && electrons_no == 2 && missingEnergy < 30"
    #select_nunu = "muons_no == 0 && electrons_no == 0 && missingEnergy > 102 && missingEnergy < 110"
    select_qq   = "muons_no == 0 && electrons_no == 0 && missingEnergy < 35"
    select_tautau =  "muons_no == 0 && electrons_no == 0 && missingEnergy > 20 && missingEnergy < 90"
    
    #df_mumu   = df.Filter(select_mumu)
    #df_ee     = df.Filter(select_ee)
    #df_nunu   = df.Filter(select_nunu)
    df_quarks = df.Filter(select_qq)
    df_tau    = df.Filter(select_tautau)
    
    #results.append(df_mumu.Histo1D(("cutFlow_mumu", "", *bins_count), "cut1"))
    #results.append(df_ee.Histo1D(("cutFlow_ee", "", *bins_count), "cut1"))
    #results.append(df_nunu.Histo1D(("cutFlow_nunu", "", *bins_count), "cut1"))
    results.append(df_quarks.Histo1D(("cutFlow_bb", "", *bins_count), "cut1"))
    results.append(df_quarks.Histo1D(("cutFlow_cc", "", *bins_count), "cut1"))
    results.append(df_quarks.Histo1D(("cutFlow_ss", "", *bins_count), "cut1"))
    results.append(df_quarks.Histo1D(("cutFlow_uu", "", *bins_count), "cut1"))
    results.append(df_quarks.Histo1D(("cutFlow_dd", "", *bins_count), "cut1"))
    results.append(df_tau.Histo1D(("cutFlow_tautau", "", *bins_count), "cut1")
    
    #########
    ### CUT 2: Define 4 jets
    #########

    
    #define 
    df_quarks = jet4Cluster.define(df_quarks)
    df_quarks = df_quarks.Define("jet_tlv", "FCCAnalyses::makeLorentzVectors(jet_px, jet_py, jet_pz, jet_e)")
    
    
    # pair jets based on distance to Z and H masses
    df_quarks = df_quarks.Define("zh_min_idx", """
            FCCAnalyses::Vec_i min{0, 0, 0, 0};
            float distm = INFINITY;
            for (int i = 0; i < 3; i++)
                for (int j = i + 1; j < 4; j++)
                    for (int k = 0; k < 3; k++) {
                        if (i == k || j == k) continue;
                        for (int l = k + 1; l < 4; l++) {
                            if (i == l || j == l) continue;
                            float distz = ((jet_tlv[i] + jet_tlv[j]).M() - 91.2);
                            float disth = ((jet_tlv[k] + jet_tlv[l]).M() - 125);
                            if (distz*distz/91.2 + disth*disth/125 < distm) {
                                distm = distz*distz + disth*disth;
                                min[0] = i; min[1] = j; min[2] = k; min[3] = l;
                            }
                        }
                    }
            return min;""")
    
    # compute Z and H masses and momenta
    df_quarks = df_quarks.Define("z_dijet", "jet_tlv[zh_min_idx[0]] + jet_tlv[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("h_dijet", "jet_tlv[zh_min_idx[2]] + jet_tlv[zh_min_idx[3]]")
    
    df_quarks = df_quarks.Define("z_dijet_m", "z_dijet.M()")
    df_quarks = df_quarks.Define("z_dijet_p", "z_dijet.P()")
    df_quarks = df_quarks.Define("h_dijet_m", "h_dijet.M()")
    df_quarks = df_quarks.Define("h_dijet_p", "h_dijet.P()")
    
    results.append(df_quarks.Histo1D(("quarks_z_m_nOne", "", *bins_m), "z_dijet_m"))
    results.append(df_quarks.Histo1D(("quarks_z_p_nOne", "", *bins_m), "z_dijet_p"))
    results.append(df_quarks.Histo1D(("quarks_h_m_nOne", "", *bins_m), "h_dijet_m"))
    results.append(df_quarks.Histo1D(("quarks_h_p_nOne", "", *bins_m), "h_dijet_p"))
    
    # filter on Z momentum
    df_quarks = df_quarks.Filter("z_dijet_p > 30 && z_dijet_p < 65")
    results.append(df_quarks.Histo1D(("cutFlow_bb", "", *bins_count), "cut2"))
    results.append(df_quarks.Histo1D(("cutFlow_cc", "", *bins_count), "cut2"))
    results.append(df_quarks.Histo1D(("cutFlow_ss", "", *bins_count), "cut2"))
    results.append(df_quarks.Histo1D(("cutFlow_uu", "", *bins_count), "cut2"))
    results.append(df_quarks.Histo1D(("cutFlow_dd", "", *bins_count), "cut2"))
    #results.append(df_quarks.Histo1D(("cutFlow_tautau", "", *bins_count), "cut2"))
    
    #########
    ### CUT 3: Higgs mass reconstruction
    #########
    
    # filter on H mass
    df_quarks = df_quarks.Filter("h_dijet_m > 100 && h_dijet_m < 150")
    results.append(df_quarks.Histo1D(("cutFlow_bb", "", *bins_count), "cut3"))
    results.append(df_quarks.Histo1D(("cutFlow_cc", "", *bins_count), "cut3"))
    results.append(df_quarks.Histo1D(("cutFlow_ss", "", *bins_count), "cut3"))
    results.append(df_quarks.Histo1D(("cutFlow_uu", "", *bins_count), "cut3"))
    results.append(df_quarks.Histo1D(("cutFlow_dd", "", *bins_count), "cut3"))
    results.append(df_quarks.Histo1D(("cutFlow_tautau", "", *bins_count), "cut2"))
    
    ########
    ### CUT 4: flavor tag
    ########
    
    # flavour tagging
    df_quarks = jet4Flavour.define_and_inference(df_quarks)
    
    # make sure that there are 2 b jets
    df_quarks = df_quarks.Define("Hbb_prob1", "recojet_isB[zh_min_idx[2]]")
    df_quarks = df_quarks.Define("Hbb_prob2", "recojet_isB[zh_min_idx[3]]")
    df_quarks = df_quarks.Define("Hbb_prob", "Hbb_prob1 + Hbb_prob2")
    
    results.append(df_quarks.Graph("Hbb_prob1", "Hbb_prob2"))
    results.append(df_quarks.Histo1D(("Hbb_prob_nOne", "", *bins_prob), "Hbb_prob"))
    
    df_quarks = df_quarks.Filter("Hbb_prob1 > 0.5 && Hbb_prob2 > 0.5")
    
    results.append(df_quarks.Histo1D(("cutFlow_bb", "", *bins_count), "cut4"))
    results.append(df_quarks.Histo1D(("cutFlow_cc", "", *bins_count), "cut4"))
    results.append(df_quarks.Histo1D(("cutFlow_ss", "", *bins_count), "cut4"))
    results.append(df_quarks.Histo1D(("cutFlow_uu", "", *bins_count), "cut4"))
    results.append(df_quarks.Histo1D(("cutFlow_dd", "", *bins_count), "cut4"))
    results.append(df_quarks.Histo1D(("cutFlow_tautau", "", *bins_count), "cut3"))
    
    # sort by tag
    df_quarks = df_quarks.Define("Zbb_prob1", "recojet_isB[zh_min_idx[0]]")
    df_quarks = df_quarks.Define("Zbb_prob2", "recojet_isB[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("Zbb_prob", "Zbb_prob1 + Zbb_prob2")
    
    df_quarks = df_quarks.Define("Zcc_prob1", "recojet_isC[zh_min_idx[0]]")
    df_quarks = df_quarks.Define("Zcc_prob2", "recojet_isC[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("Zcc_prob", "Zcc_prob1 + Zcc_prob2")
    
    df_quarks = df_quarks.Define("Zss_prob1", "recojet_isS[zh_min_idx[0]]")
    df_quarks = df_quarks.Define("Zss_prob2", "recojet_isS[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("Zss_prob", "Zss_prob1 + Zss_prob2")
    
    df_quarks = df_quarks.Define("Zuu_prob1", "recojet_isU[zh_min_idx[0]]")
    df_quarks = df_quarks.Define("Zuu_prob2", "recojet_isU[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("Zuu_prob", "Zuu_prob1 + Zuu_prob2")

    df_quarks = df_quarks.Define("Zdd_prob1", "recojet_isD[zh_min_idx[0]]")
    df_quarks = df_quarks.Define("Zdd_prob2", "recojet_isD[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("Zdd_prob", "Zdd_prob1 + Zdd_prob2")
    
    df_quarks = df_quarks.Define("Ztautau_prob1", "recojet_isTAU[zh_min_idx[0]]")
    df_quarks = df_quarks.Define("Ztautau_prob2", "recojet_isTAU[zh_min_idx[1]]")
    df_quarks = df_quarks.Define("Ztautau_prob", "Ztautau_prob1 + Ztautau_prob2")
    
    results.append(df_quarks.Graph("Zbb_prob1", "Zbb_prob2"))
    results.append(df_quarks.Graph("Zcc_prob1", "Zcc_prob2"))
    results.append(df_quarks.Graph("Zss_prob1", "Zss_prob2"))
    results.append(df_quarks.Graph("Zuu_prob1", "Zuu_prob2"))
    results.append(df_quarks.Graph("Zdd_prob1", "Zdd_prob2"))
    results.append(df_quarks.Graph("Ztautau_prob1", "Ztautau_prob2"))
    
    results.append(df_quarks.Histo1D(("Zbb_prob_nOne", "", *bins_prob), "Zbb_prob"))
    results.append(df_quarks.Histo1D(("Zcc_prob_nOne", "", *bins_prob), "Zcc_prob"))
    results.append(df_quarks.Histo1D(("Zss_prob_nOne", "", *bins_prob), "Zss_prob"))
    results.append(df_quarks.Histo1D(("Zuu_prob_nOne", "", *bins_prob), "Zuu_prob"))
    results.append(df_quarks.Histo1D(("Zdd_prob_nOne", "", *bins_prob), "Zdd_prob"))
    results.append(df_quarks.Histo1D(("Ztautau_prob_nOne", "", *bins_prob), "Ztautau_prob"))
    
    df_quarks = df_quarks.Define("best_tag", """
            if (Zbb_prob > Zcc_prob && Zbb_prob > Zss_prob && Zbb_prob > Zuu_prob && Zbb_prob > Zdd_prob && Zbb_prob > Ztautau_prob) {
                return 0;
            } else if (Zcc_prob > Zss_prob && Zcc_prob > Zuu_prob > Zdd_prob > Ztautau_prob) {
                return 1;
            } else if (Zss_prob > Zuu_prob > Zdd_prob> Ztautau_prob) {
                return 2;
            } else if (Zuu_prob > Zdd_prob > Ztautau_prob) {
                return 3;
            } else if (Zdd_prob > Ztautau_prob) {
                return 4;
            } else {
                return 5;
            } """)
    
    df_bb = df_quarks.Filter("best_tag == 0 && Zbb_prob > 2*0.5")
    df_cc = df_quarks.Filter("best_tag == 1 && Zcc_prob > 1.6")
    df_ss = df_quarks.Filter("best_tag == 2 && Zss_prob > 1.2")
    df_uu = df_quarks.Filter("best_tag == 3 && Zuu_prob > 1.1")
    df_dd = df_quarks.Filter("best_tag == 3 && Zdd_prob > 1.1")
    df_tautau = df_quarks.Filter("best_tag == 4 && Ztautau_prob > 2*0.1")
    
    results.append(df_bb.Histo1D(("cutFlow_bb", "", *bins_count), "cut5"))
    results.append(df_cc.Histo1D(("cutFlow_cc", "", *bins_count), "cut5"))
    results.append(df_ss.Histo1D(("cutFlow_ss", "", *bins_count), "cut5"))
    results.append(df_uu.Histo1D(("cutFlow_uu", "", *bins_count), "cut5"))
    results.append(df_dd.Histo1D(("cutFlow_dd", "", *bins_count), "cut5"))
    results.append(df_tautau.Histo1D(("cutFlow_tautau", "", *bins_count), "cut4"))
    
    
    # make final mass and momentum histograms
    for q, df in [("bb", df_bb), ("cc", df_cc), ("ss", df_ss), ("qq", df_uu),("qq", df_dd),("tautau", df_tautau)]:
        results.append(df.Histo1D((f"z{q}_z_m", "", *bins_m), "z_dijet_m"))
        results.append(df.Histo1D((f"z{q}_h_m", "", *bins_m), "h_dijet_m"))
        results.append(df.Histo1D((f"z{q}_z_p", "", *bins_m), "z_dijet_p"))
        results.append(df.Histo1D((f"z{q}_h_p", "", *bins_m), "h_dijet_p"))

    
    return results, weightsum


if __name__ == "__main__":

    datadict = functions.get_datadicts() # get default datasets

    Zprods = ["ee", "mumu", "tautau", "nunu", "qq", "ss", "cc", "bb"] 
    bb_sig = [f"wzp6_ee_{i}H_Hbb_ecm240" for i in Zprods]
    cc_sig = [f"wzp6_ee_{i}H_Hcc_ecm240" for i in Zprods]
    gg_sig = [f"wzp6_ee_{i}H_Hgg_ecm240" for i in Zprods]
    
    datasets_bkg=["p8_ee_WW_ecm240","p8_ee_ZZ_ecm240","wzp6_ee_mumu_ecm240", "wzp6_ee_tautau_ecm240", "wzp6_egamma_eZ_Zmumu_ecm240", "wzp6_gammae_eZ_Zmumu_ecm240", "wzp6_gaga_mumu_60_ecm240", "wzp6_gaga_tautau_60_ecm240"]

    datasets_to_run = bb_sig + cc_sig + gg_sig+ datasets_bkg
    
    result = functions.build_and_run(datadict, datasets_to_run, build_graph, f"output_h_bb_Ztautau.root", args,norm=True, lumi=7200000)
