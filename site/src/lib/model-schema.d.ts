/* eslint-disable */
/**
 * This file was automatically generated by json-schema-to-typescript.
 * DO NOT MODIFY IT BY HAND. Instead, modify the source JSONSchema file,
 * and run json-schema-to-typescript to regenerate this file.
 */

export interface ModelMetadata {
  model_name: string
  model_version: string
  matbench_discovery_version: string
  date_added: string
  date_published: string
  authors: {
    name: string
    affiliation?: string
    email?: string
    orcid?: string
    [k: string]: unknown
  }[]
  trained_by?: {
    name: string
    affiliation?: string
    orcid?: string
    github?: string
    [k: string]: unknown
  }[]
  repo: string
  doi: string
  paper: string
  url?: string
  pypi?: string
  requirements: {
    /**
     * This interface was referenced by `undefined`'s JSON-Schema definition
     * via the `patternProperty` "^[a-zA-Z]{1}[a-zA-Z0-9_\-]{0,}$".
     */
    [k: string]: string
  }
  trained_for_benchmark: boolean
  training_set: (
    | 'MP 2022'
    | 'MPtrj'
    | 'MPF'
    | 'MP Graphs'
    | 'GNoME'
    | 'MatterSim'
    | 'Alex'
    | 'OMat24'
    | 'sAlex'
  )[]
  hyperparams?: {
    max_force?: number
    max_steps?: number
    optimizer?: string
    ase_optimizer?: string
    learning_rate?: number
    batch_size?: number
    epochs?: number
    n_layers?: number
    radial_cutoff?: number
    [k: string]: unknown
  }
  notes?: {
    Description?: string
    Training?:
      | string
      | {
          [k: string]: string
        }
    'Missing Preds'?: string
    [k: string]: unknown
  }
  model_params: number
  n_estimators: number
  train_task: 'RP2RE' | 'RS2RE' | 'S2E' | 'S2RE' | 'S2EF' | 'S2EFS' | 'S2EFSM'
  test_task: 'IP2E' | 'IS2E' | 'IS2RE' | 'IS2RE-SR' | 'IS2RE-BO'
  model_type: 'GNN' | 'UIP' | 'BO-GNN' | 'Fingerprint' | 'Transformer' | 'RF'
  targets: 'E' | 'EF' | 'EFS' | 'EFSM'
  openness?: 'OSOD' | 'OSCD' | 'CSOD' | 'CSCD'
  pred_col: string
  status?: 'aborted' | 'complete'
}
