pod=$1
shift
kubectl exec -i $pod -- "$@"
