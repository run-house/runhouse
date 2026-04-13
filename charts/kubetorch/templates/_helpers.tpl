{{/*
Generate pod anti-affinity rules for a component.

Usage:
  {{ include "kubetorch.podAntiAffinity" (dict "component" "kubetorch-controller" "config" .Values.kubetorchController.podAntiAffinity) }}

Parameters:
  .component - the app label value (e.g. "kubetorch-controller")
  .config    - the podAntiAffinity config block from values
*/}}
{{- define "kubetorch.podAntiAffinity" -}}
{{- if .config.enabled }}
podAntiAffinity:
  {{- if eq .config.type "required" }}
  requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
          - key: app
            operator: In
            values:
              - {{ .component }}
      topologyKey: {{ .config.topologyKey | default "kubernetes.io/hostname" }}
    {{- if .config.zoneSpread }}
    - labelSelector:
        matchExpressions:
          - key: app
            operator: In
            values:
              - {{ .component }}
      topologyKey: topology.kubernetes.io/zone
    {{- end }}
  {{- else }}
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
            - key: app
              operator: In
              values:
                - {{ .component }}
        topologyKey: {{ .config.topologyKey | default "kubernetes.io/hostname" }}
    {{- if .config.zoneSpread }}
    - weight: 50
      podAffinityTerm:
        labelSelector:
          matchExpressions:
            - key: app
              operator: In
              values:
                - {{ .component }}
        topologyKey: topology.kubernetes.io/zone
    {{- end }}
  {{- end }}
{{- end }}
{{- end -}}
