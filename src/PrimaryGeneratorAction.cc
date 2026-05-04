#include "PrimaryGeneratorAction.hh"

#include "ExperimentConfig.hh"

#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4Gamma.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4ios.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

PrimaryGeneratorAction::PrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction(),
  fParticleGun(new G4ParticleGun(1)),
  fSourceMode(SourceMode::Spectrum),
  fMonoEnergy_keV(80.0),
  fSourcePos(-30.0 * cm, 0.0, 0.0),
  fBeamDir(1.0, 0.0, 0.0),
  fBeamHalfY_mm(5.0),
  fBeamHalfZ_mm(5.0)
{
  auto particleDefinition = G4Gamma::GammaDefinition();
  fParticleGun->SetParticleDefinition(particleDefinition);

  const auto& config = GetExperimentConfig();

  fSourceMode = config.sourceMode;
  fMonoEnergy_keV = config.monoEnergyKeV;
  fSourcePos = G4ThreeVector(config.sourceX_cm * cm,
                             config.sourceY_mm * mm,
                             config.sourceZ_mm * mm);
  fBeamDir = G4ThreeVector(config.dirX, config.dirY, config.dirZ);
  fBeamHalfY_mm = config.beamHalfY_mm;
  fBeamHalfZ_mm = config.beamHalfZ_mm;

  if (fSourceMode == SourceMode::Mono) {
    G4cout << "[SourceConfig] mode = mono" << G4endl;
    G4cout << "[SourceConfig] mono energy = " << fMonoEnergy_keV << " keV"
           << G4endl;
  }

  if (fSourceMode == SourceMode::Spectrum) {
    LoadSpectrumFile(config.spectrumFile);
    G4cout << "[SourceConfig] mode = spectrum" << G4endl;
    G4cout << "[SourceConfig] spectrum file = " << config.spectrumFile
           << G4endl;
  }

  if (fSourceMode == SourceMode::PhaseSpace) {
    LoadPhaseSpaceFile(config.phaseSpaceFile);
    G4cout << "[SourceConfig] mode = phase_space" << G4endl;
    G4cout << "[SourceConfig] phase-space file = " << config.phaseSpaceFile
           << G4endl;
  }

  if (fSourceMode == SourceMode::Spectrum && fSpectrumCDF.empty()) {
    G4Exception("PrimaryGeneratorAction::PrimaryGeneratorAction()",
                "EmptySpectrum",
                FatalException,
                "No spectrum was loaded.");
  }

  if (fSourceMode == SourceMode::PhaseSpace && fSamples.empty()) {
    G4Exception("PrimaryGeneratorAction::PrimaryGeneratorAction()",
                "EmptyPhaseSpace",
                FatalException,
                "No photon samples were loaded from phase-space file.");
  }
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::LoadSpectrumFile(const G4String& fileName)
{
  std::ifstream fin(fileName);
  if (!fin) {
    G4Exception("PrimaryGeneratorAction::LoadSpectrumFile()",
                "SpectrumFileNotFound",
                FatalException,
                ("Cannot open spectrum file: " + std::string(fileName)).c_str());
  }

  std::string line;
  std::getline(fin, line); // skip header

  G4double cumulative = 0.0;

  while (std::getline(fin, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string token;

    G4double energy_keV = 0.0;
    G4double weight = 0.0;

    std::getline(ss, token, ',');
    energy_keV = std::stod(token);

    std::getline(ss, token, ',');
    weight = std::stod(token);

    if (weight <= 0.0) continue;

    fSpectrumEnergies_keV.push_back(energy_keV);
    cumulative += weight;
    fSpectrumCDF.push_back(cumulative);
  }

  if (!fSpectrumCDF.empty()) {
    for (auto& v : fSpectrumCDF) {
      v /= cumulative;
    }
  }

  G4cout << "[Spectrum] Loaded " << fSpectrumEnergies_keV.size()
         << " energy bins from " << fileName << G4endl;
}

void PrimaryGeneratorAction::LoadPhaseSpaceFile(const G4String& fileName)
{
  std::ifstream fin(fileName);
  if (!fin) {
    G4Exception("PrimaryGeneratorAction::LoadPhaseSpaceFile()",
                "PhaseSpaceFileNotFound",
                FatalException,
                ("Cannot open phase-space file: " + std::string(fileName)).c_str());
  }

  std::string line;
  std::getline(fin, line); // CSV header
  std::vector<std::string> columns;
  {
    std::stringstream header(line);
    std::string token;
    while (std::getline(header, token, ',')) {
      token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
      columns.push_back(token);
    }
  }
  const auto hasXmm = std::find(columns.begin(), columns.end(), "x_mm") != columns.end();

  while (std::getline(fin, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string token;
    SourcePhotonSample sample{};
    sample.x_mm = 0.0;

    std::getline(ss, token, ','); // event_id skip

    std::getline(ss, token, ',');
    sample.energy_keV = std::stod(token);

    if (hasXmm) {
      std::getline(ss, token, ',');
      sample.x_mm = std::stod(token);
    }

    std::getline(ss, token, ',');
    sample.y_mm = std::stod(token);

    std::getline(ss, token, ',');
    sample.z_mm = std::stod(token);

    std::getline(ss, token, ',');
    sample.dir_x = std::stod(token);

    std::getline(ss, token, ',');
    sample.dir_y = std::stod(token);

    std::getline(ss, token, ',');
    sample.dir_z = std::stod(token);

    fSamples.push_back(sample);
  }

  G4cout << "[PhaseSpace] Loaded " << fSamples.size()
         << " source photons from " << fileName << G4endl;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  if (fSourceMode == SourceMode::Mono) {
    G4double y =
        fSourcePos.y() + (2.0 * G4UniformRand() - 1.0) * fBeamHalfY_mm * mm;
    G4double z =
        fSourcePos.z() + (2.0 * G4UniformRand() - 1.0) * fBeamHalfZ_mm * mm;

    fParticleGun->SetParticlePosition(G4ThreeVector(fSourcePos.x(), y, z));
    fParticleGun->SetParticleMomentumDirection(fBeamDir.unit());
    fParticleGun->SetParticleEnergy(fMonoEnergy_keV * keV);
  }
  else if (fSourceMode == SourceMode::Spectrum) {
    auto u = G4UniformRand();
    auto it = std::lower_bound(fSpectrumCDF.begin(), fSpectrumCDF.end(), u);
    auto idx = std::distance(fSpectrumCDF.begin(), it);
    if (idx >= (G4int)fSpectrumEnergies_keV.size()) {
      idx = fSpectrumEnergies_keV.size() - 1;
    }

    G4double energy = fSpectrumEnergies_keV[idx] * keV;

    G4double y = fSourcePos.y() + (2.0 * G4UniformRand() - 1.0) * fBeamHalfY_mm * mm;
    G4double z = fSourcePos.z() + (2.0 * G4UniformRand() - 1.0) * fBeamHalfZ_mm * mm;

    fParticleGun->SetParticlePosition(G4ThreeVector(fSourcePos.x(), y, z));
    fParticleGun->SetParticleMomentumDirection(fBeamDir.unit());
    fParticleGun->SetParticleEnergy(energy);
  }
  else {
    auto idx = G4RandFlat::shootInt((G4int)fSamples.size());
    const auto& s = fSamples[idx];

    fParticleGun->SetParticlePosition(
      G4ThreeVector(fSourcePos.x() + s.x_mm * mm, s.y_mm * mm, s.z_mm * mm));

    fParticleGun->SetParticleMomentumDirection(
      G4ThreeVector(s.dir_x, s.dir_y, s.dir_z).unit());

    fParticleGun->SetParticleEnergy(s.energy_keV * keV);
  }

  fParticleGun->GeneratePrimaryVertex(anEvent);
}
