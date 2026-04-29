#include "RunAction.hh"

#include "ExperimentConfig.hh"

#include "G4AccumulableManager.hh"
#include "G4AutoLock.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace
{
G4Mutex csvMutex = G4MUTEX_INITIALIZER;
}

namespace B1
{

RunAction::RunAction()
: G4UserRunAction(),
  fEventCsvFileName(BuildOutputPath(GetExperimentConfig(), "_events.csv")),
  fHitCsvFileName(BuildOutputPath(GetExperimentConfig(), "_hits.csv")),
  fMetadataFileName(BuildOutputPath(GetExperimentConfig(), "_metadata.json")),
  fSumDetectorEdep(0.),
  fSumDetectorGammaEntries(0),
  fSumPrimaryGammaEntries(0)
{
  auto* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Register(fSumDetectorEdep);
  accumulableManager->Register(fSumDetectorGammaEntries);
  accumulableManager->Register(fSumPrimaryGammaEntries);
}

RunAction::~RunAction() = default;

void RunAction::BeginOfRunAction(const G4Run*)
{
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);
  G4AccumulableManager::Instance()->Reset();

  std::filesystem::create_directories(
      std::filesystem::path(fEventCsvFileName).parent_path());
  std::filesystem::create_directories(
      std::filesystem::path(fHitCsvFileName).parent_path());

  if (IsMaster()) {
    {
      std::ofstream outFile(fEventCsvFileName, std::ios::out);
      outFile << "event_id,detector_edep_keV,detector_gamma_entries,primary_gamma_entries,"
              << "transmission_gamma_entries,transmission_primary_gamma_entries,"
              << "side_scatter_gamma_entries,side_scatter_primary_gamma_entries\n";
    }
    {
      std::ofstream outFile(fHitCsvFileName, std::ios::out);
      outFile << "event_id,detector_id,x_mm,y_mm,z_mm,photon_energy_keV,is_primary,theta_deg,is_direct_primary,is_scattered_primary\n";
    }
  }
}

void RunAction::AddRunDetectorEdep(G4double edep)
{
  fSumDetectorEdep += edep;
}

void RunAction::AddRunDetectorGammaEntries(G4int n)
{
  fSumDetectorGammaEntries += n;
}

void RunAction::AddRunPrimaryGammaEntries(G4int n)
{
  fSumPrimaryGammaEntries += n;
}

void RunAction::WriteEventData(G4int eventID,
                               G4double detectorEdep_keV,
                               G4int detectorGammaEntries,
                               G4int primaryGammaEntries,
                               G4int transmissionGammaEntries,
                               G4int transmissionPrimaryGammaEntries,
                               G4int sideScatterGammaEntries,
                               G4int sideScatterPrimaryGammaEntries)
{
  G4AutoLock lock(&csvMutex);

  std::ofstream outFile(fEventCsvFileName, std::ios::app);
  outFile << eventID << ","
          << std::fixed << std::setprecision(6) << detectorEdep_keV << ","
          << detectorGammaEntries << ","
          << primaryGammaEntries << ","
          << transmissionGammaEntries << ","
          << transmissionPrimaryGammaEntries << ","
          << sideScatterGammaEntries << ","
          << sideScatterPrimaryGammaEntries << "\n";
}

void RunAction::WriteHitData(G4int eventID,
                             const std::string& detectorId,
                             G4double x_mm,
                             G4double y_mm,
                             G4double z_mm,
                             G4double photonEnergy_keV,
                             G4bool isPrimary,
                             G4double theta_deg,
                             G4bool isDirectPrimary,
                             G4bool isScatteredPrimary)
{
  G4AutoLock lock(&csvMutex);

  std::ofstream outFile(fHitCsvFileName, std::ios::app);
  outFile << eventID << ","
          << detectorId << ","
          << std::fixed << std::setprecision(6)
          << x_mm << ","
          << y_mm << ","
          << z_mm << ","
          << photonEnergy_keV << ","
          << (isPrimary ? 1 : 0) << ","
          << theta_deg << ","
          << (isDirectPrimary ? 1 : 0) << ","
          << (isScatteredPrimary ? 1 : 0) << "\n";
}

void RunAction::WriteMetadataFile(G4int nofEvents,
                                  G4double totalEdep_keV,
                                  G4double meanEdep_keV,
                                  G4int totalGammaEntries,
                                  G4int totalPrimaryEntries) const
{
  const auto& config = GetExperimentConfig();
  std::ofstream outFile(fMetadataFileName, std::ios::out);
  outFile << "{\n";
  outFile << "  \"run_id\": \"" << config.runId << "\",\n";
  outFile << "  \"experiment_label\": \"" << config.experimentLabel << "\",\n";
  outFile << "  \"output_prefix\": \"" << config.outputPrefix << "\",\n";
  outFile << "  \"output_dir\": \"" << config.outputDir << "\",\n";
  outFile << "  \"benchmark_suite\": \"" << config.benchmarkSuite << "\",\n";
  outFile << "  \"research_route\": \"" << config.researchRoute << "\",\n";
  outFile << "  \"prediction_stage\": \"" << config.predictionStage << "\",\n";
  outFile << "  \"run_role\": \"" << config.runRole << "\",\n";
  outFile << "  \"source_variant\": \"" << config.sourceVariant << "\",\n";
  outFile << "  \"detector_layout\": \"" << config.detectorLayout << "\",\n";
  outFile << "  \"prep_profile\": \"" << config.prepProfile << "\",\n";
  outFile << "  \"feed_size_band\": \"" << config.feedSizeBand << "\",\n";
  outFile << "  \"feed_condition\": \"" << config.feedCondition << "\",\n";
  outFile << "  \"config_path\": \"" << config.configPath << "\",\n";
  outFile << "  \"source_mode\": \"" << SourceModeToString(config.sourceMode)
          << "\",\n";
  outFile << "  \"spectrum_file\": \"" << config.spectrumFile << "\",\n";
  outFile << "  \"phase_space_file\": \"" << config.phaseSpaceFile << "\",\n";
  outFile << "  \"mono_energy_keV\": " << config.monoEnergyKeV << ",\n";
  outFile << "  \"source_x_cm\": " << config.sourceX_cm << ",\n";
  outFile << "  \"source_y_mm\": " << config.sourceY_mm << ",\n";
  outFile << "  \"source_z_mm\": " << config.sourceZ_mm << ",\n";
  outFile << "  \"beam_half_y_mm\": " << config.beamHalfY_mm << ",\n";
  outFile << "  \"beam_half_z_mm\": " << config.beamHalfZ_mm << ",\n";
  outFile << "  \"dir_x\": " << config.dirX << ",\n";
  outFile << "  \"dir_y\": " << config.dirY << ",\n";
  outFile << "  \"dir_z\": " << config.dirZ << ",\n";
  outFile << "  \"incidence_angle_deg\": " << config.incidenceAngleDeg << ",\n";
  outFile << "  \"ore_material_mode\": \""
          << OreMaterialModeToString(config.oreMaterialMode) << "\",\n";
  outFile << "  \"ore_primary_material\": \"" << config.orePrimaryMaterial
          << "\",\n";
  outFile << "  \"ore_secondary_material\": \""
          << config.oreSecondaryMaterial << "\",\n";
  outFile << "  \"ore_secondary_mass_fraction\": "
          << config.oreSecondaryMassFraction << ",\n";
  outFile << "  \"ore_shape\": \"" << OreShapeToString(config.oreShape)
          << "\",\n";
  outFile << "  \"ore_thickness_mm\": " << config.oreThickness_mm << ",\n";
  outFile << "  \"ore_half_y_mm\": " << config.oreHalfY_mm << ",\n";
  outFile << "  \"ore_half_z_mm\": " << config.oreHalfZ_mm << ",\n";
  outFile << "  \"heterogeneity_mode\": \""
          << HeterogeneityModeToString(config.heterogeneityMode) << "\",\n";
  outFile << "  \"inclusion_material\": \"" << config.inclusionMaterial
          << "\",\n";
  outFile << "  \"inclusion_shape\": \"" << OreShapeToString(config.inclusionShape)
          << "\",\n";
  outFile << "  \"inclusion_thickness_mm\": " << config.inclusionThickness_mm
          << ",\n";
  outFile << "  \"inclusion_radius_y_mm\": " << config.inclusionRadiusY_mm
          << ",\n";
  outFile << "  \"inclusion_radius_z_mm\": " << config.inclusionRadiusZ_mm
          << ",\n";
  outFile << "  \"inclusion_offset_y_mm\": " << config.inclusionOffsetY_mm
          << ",\n";
  outFile << "  \"inclusion_offset_z_mm\": " << config.inclusionOffsetZ_mm
          << ",\n";
  outFile << "  \"detector_thickness_mm\": " << config.detectorThickness_mm
          << ",\n";
  outFile << "  \"detector_half_y_mm\": " << config.detectorHalfY_mm << ",\n";
  outFile << "  \"detector_half_z_mm\": " << config.detectorHalfZ_mm << ",\n";
  outFile << "  \"detector_x_cm\": " << config.detectorX_cm << ",\n";
  outFile << "  \"side_detector_thickness_mm\": "
          << config.sideDetectorThickness_mm << ",\n";
  outFile << "  \"side_detector_half_x_mm\": "
          << config.sideDetectorHalfX_mm << ",\n";
  outFile << "  \"side_detector_half_z_mm\": "
          << config.sideDetectorHalfZ_mm << ",\n";
  outFile << "  \"side_detector_y_cm\": " << config.sideDetectorY_cm << ",\n";
  outFile << "  \"sample_photons\": " << config.samplePhotons << ",\n";
  outFile << "  \"random_seed\": " << config.randomSeed << ",\n";
  outFile << "  \"event_file\": \"" << fEventCsvFileName << "\",\n";
  outFile << "  \"hit_file\": \"" << fHitCsvFileName << "\",\n";
  outFile << "  \"n_events\": " << nofEvents << ",\n";
  outFile << "  \"total_detector_edep_keV\": " << totalEdep_keV << ",\n";
  outFile << "  \"mean_detector_edep_keV\": " << meanEdep_keV << ",\n";
  outFile << "  \"total_detector_gamma_entries\": " << totalGammaEntries
          << ",\n";
  outFile << "  \"total_primary_gamma_entries\": " << totalPrimaryEntries
          << "\n";
  outFile << "}\n";
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  G4int nofEvents = run->GetNumberOfEvent();
  if (nofEvents == 0) return;

  G4AccumulableManager::Instance()->Merge();

  G4double totalEdep = fSumDetectorEdep.GetValue();
  G4int totalGammaEntries = fSumDetectorGammaEntries.GetValue();
  G4int totalPrimaryEntries = fSumPrimaryGammaEntries.GetValue();

  G4double meanEdep = totalEdep / nofEvents;
  WriteMetadataFile(nofEvents,
                    totalEdep / keV,
                    meanEdep / keV,
                    totalGammaEntries,
                    totalPrimaryEntries);

  G4cout << "\n================ XRT Research Run Summary ================\n"
         << "Number of events             : " << nofEvents << G4endl
         << "Total detector edep          : " << totalEdep / keV << " keV" << G4endl
         << "Mean detector edep / event   : " << meanEdep / keV << " keV" << G4endl
         << "Total detector gamma entries : " << totalGammaEntries << G4endl
         << "Primary gamma entries        : " << totalPrimaryEntries << G4endl
         << "Event CSV output file        : " << fEventCsvFileName << G4endl
         << "Hit CSV output file          : " << fHitCsvFileName << G4endl
         << "Metadata file               : " << fMetadataFileName << G4endl
         << "=============================================================\n"
         << G4endl;
}

}
